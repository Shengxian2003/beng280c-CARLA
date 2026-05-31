"""Skill-as-tool wrappers for the Stage 3 agent.

Each tool exposes one Stage-2 skill to the LLM as a JSON-schema-described
function. The LLM never sees raw ndarrays — those live in a per-session
``Workspace`` keyed by human-readable names the LLM itself chooses.

    from utility.tools import Workspace, TOOLS, call_tool, tools_prompt_block

    ws = Workspace()
    result = call_tool(ws, "load_reconstruction",
                       {"mat_path": "/mnt/g/medict_tmp/recon_cs_5iter.mat"})

The workspace also collects intermediate outputs (suggested seeds, named
masks, verifier verdicts) so the agent can refer back to them by name in
later turns rather than re-computing.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from skills.reconstruction import reconstruct as _reconstruct
from skills.segmentation import (
    suggest_seed_points as _suggest_seed_points,
    segment_from_seed as _segment_from_seed,
)
from skills.physics_verifier import verify as _verify
from skills.hemodynamic import analyze as _analyze

from .session_store import SessionStore, is_path_allowed


# ============================================================================
# Workspace — agent's in-memory state across tool calls
# ============================================================================

@dataclass
class Workspace:
    """Holds the things tools need to share — none of which the LLM should see directly."""

    # Reconstruction output (xHat, thetaX/Y/Z, meta, ...)
    recon: dict | None = None

    # Named vessel masks the LLM has segmented this session
    masks: dict[str, np.ndarray] = field(default_factory=dict)

    # Cached PC-MRA seed candidates (LLM can ask once, then refer back)
    suggested_seeds: list[dict] | None = None

    # Per-mask verdicts and hemodynamic reports
    verdicts: dict[str, dict] = field(default_factory=dict)
    analyses: dict[str, dict] = field(default_factory=dict)

    # Session config used by tools that need it
    venc_m_per_s: float = 1.5
    voxel_size_mm: tuple[float, float, float] = (2.0, 2.0, 2.0)
    dt_seconds: float = 0.05

    # On-disk artifact store (set by the orchestrator at session start).
    # Tools call store.write_* in addition to mutating the workspace so the
    # session directory always has a grounded, LLM-readable snapshot.
    store: Optional[SessionStore] = None

    # The specialist role currently driving tool calls. Set by Specialist
    # at the start of each handle() so the scoped read_file tool can check
    # the caller's permissions against READ_SCOPES.
    current_role: str = ""

    def require_recon(self) -> dict:
        if self.recon is None:
            raise ToolError("no reconstruction in workspace — call reconstruct or load_reconstruction first")
        return self.recon

    def require_mask(self, name: str) -> np.ndarray:
        if name not in self.masks:
            known = sorted(self.masks.keys()) or "<none>"
            raise ToolError(f"mask {name!r} not found — known masks: {known}")
        return self.masks[name]


# ============================================================================
# Errors + validation
# ============================================================================

class ToolError(ValueError):
    """Raised by a tool to signal a recoverable error the LLM can react to."""


def _validate_args(args: dict, schema: dict) -> dict:
    """Minimal JSON-schema check: type, enum, minimum, maximum, required + defaults.

    Returns a new dict with defaults applied. Raises ToolError on violation —
    the orchestrator turns that into a structured response the LLM can read.
    """
    if schema.get("type") != "object":
        raise ValueError(f"top-level schema must be an object, got {schema.get('type')}")

    props: dict = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    out = {}
    # Apply defaults + check required
    for key, pschema in props.items():
        if key in args:
            out[key] = args[key]
        elif "default" in pschema:
            out[key] = pschema["default"]
        elif key in required:
            raise ToolError(f"missing required argument {key!r}")

    # Reject unknown keys — better to fail loudly than silently drop typos
    unknown = set(args) - set(props)
    if unknown:
        raise ToolError(f"unknown argument(s): {sorted(unknown)}; expected one of {sorted(props)}")

    # Type / enum / range checks per property
    for key, val in out.items():
        pschema = props[key]
        _check_value(key, val, pschema)
    return out


_TYPE_MAP = {
    "string":  (str,),
    "integer": (int,),
    "number":  (int, float),
    "boolean": (bool,),
    "array":   (list, tuple),
    "object":  (dict,),
    "null":    (type(None),),
}


def _check_value(key: str, val: Any, pschema: dict):
    expected = pschema.get("type")
    if expected is not None:
        types = _TYPE_MAP.get(expected)
        if types is None:
            raise ValueError(f"unknown JSON-schema type {expected!r}")
        # bool is a subclass of int in Python — handle the common confusion explicitly
        if expected == "integer" and isinstance(val, bool):
            raise ToolError(f"{key}: expected integer, got bool")
        if not isinstance(val, types):
            raise ToolError(f"{key}: expected {expected}, got {type(val).__name__}")

    if "enum" in pschema and val not in pschema["enum"]:
        raise ToolError(f"{key}: must be one of {pschema['enum']}, got {val!r}")
    if "minimum" in pschema and val < pschema["minimum"]:
        raise ToolError(f"{key}: {val} below minimum {pschema['minimum']}")
    if "maximum" in pschema and val > pschema["maximum"]:
        raise ToolError(f"{key}: {val} above maximum {pschema['maximum']}")


# ============================================================================
# JSON sanitization — strip numpy / clip large arrays before returning to LLM
# ============================================================================

_MAX_INLINE_ARRAY = 64  # items above this get summarized, not serialized


def to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        if obj.size > _MAX_INLINE_ARRAY:
            return f"<ndarray shape={list(obj.shape)} dtype={obj.dtype}>"
        return obj.tolist()
    if isinstance(obj, (np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.complexfloating, complex)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    return obj


# ============================================================================
# Tool spec + implementations
# ============================================================================

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict                 # JSON Schema for arguments
    function: Callable[..., dict]    # (workspace, **validated_args) -> dict


# ----- load_reconstruction (fast — for agent dev without re-running MATLAB) -

def _tool_load_reconstruction(ws: Workspace, mat_path: str,
                              venc_m_per_s: float = 1.5,
                              voxel_size_mm_dz: float = 2.0,
                              voxel_size_mm_dy: float = 2.0,
                              voxel_size_mm_dx: float = 2.0) -> dict:
    import scipy.io as sio
    raw = sio.loadmat(mat_path, squeeze_me=True)
    if "outputs" not in raw:
        raise ToolError(f"{mat_path} has no 'outputs' struct — not a Stage 2a recon output")
    o = raw["outputs"]
    ws.recon = {
        "xHat":   np.array(o["xHat"].item()),
        "thetaX": np.array(o["thetaX"].item()),
        "thetaY": np.array(o["thetaY"].item()),
        "thetaZ": np.array(o["thetaZ"].item()),
    }
    ws.venc_m_per_s = venc_m_per_s
    ws.voxel_size_mm = (voxel_size_mm_dz, voxel_size_mm_dy, voxel_size_mm_dx)
    ws.suggested_seeds = None  # invalidate cache
    info = {
        "status": "loaded",
        "source": "real_scan",
        "source_path": mat_path,
        "shape_ZYXT": list(ws.recon["xHat"].shape),
        "venc_m_per_s": venc_m_per_s,
        "voxel_size_mm": list(ws.voxel_size_mm),
    }
    _persist_recon(ws, info)
    return info


def _persist_recon(ws: "Workspace", summary: dict) -> None:
    """Dual-write: dump recon summary JSON + raw arrays to the session store."""
    if ws.store is None or ws.recon is None:
        return
    arrays = {k: np.asarray(ws.recon[k])
              for k in ("xHat", "thetaX", "thetaY", "thetaZ")
              if k in ws.recon}
    ws.store.write_recon(summary=summary, arrays=arrays)


# ----- reconstruct (slow — wraps Stage 2a) ---------------------------------

def _tool_reconstruct(ws: Workspace, kspace_path: str,
                      method: str = "cs",
                      n_iterations: int = 50,
                      venc_m_per_s: float = 1.5,
                      voxel_size_mm_dz: float = 2.0,
                      voxel_size_mm_dy: float = 2.0,
                      voxel_size_mm_dx: float = 2.0,
                      use_gpu: bool = True) -> dict:
    """Run a full CS/CORe reconstruction. Shows a live elapsed-time counter
    + latest MATLAB output line every ``_RECON_PROGRESS_INTERVAL_S`` seconds
    so the user knows the system isn't hung during the 1-12 min MATLAB call."""
    import sys as _sys
    import threading as _threading

    t0 = time.time()
    latest_line = [""]                  # mutable holder shared with the timer thread
    stop_event = _threading.Event()

    def _capture_line(line: str) -> None:
        # Keep the most recent non-empty line so the status preview shows
        # the latest MATLAB iteration message, not just "starting…".
        if line.strip():
            latest_line[0] = line.strip()

    def _print_progress() -> None:
        while not stop_event.wait(_RECON_PROGRESS_INTERVAL_S):
            elapsed = time.time() - t0
            mins, secs = divmod(int(elapsed), 60)
            line_preview = latest_line[0][:80] if latest_line[0] else "(MATLAB warming up...)"
            # \r + flush so multiple status updates overwrite cleanly on one line
            _sys.stderr.write(
                f"\r[recon] {mins:>2d}m {secs:>2d}s elapsed — {line_preview:<80s}"
            )
            _sys.stderr.flush()

    _sys.stderr.write(
        f"\n[recon] starting MATLAB ({method}, {n_iterations} iter — expect "
        f"{'~12 min on RTX 5090' if n_iterations >= 50 else '~1-3 min'})\n"
    )
    _sys.stderr.flush()

    timer = _threading.Thread(target=_print_progress, daemon=True)
    timer.start()
    try:
        result = _reconstruct(
            kspace_path,
            method=method,
            n_iterations=n_iterations,
            venc_m_per_s=venc_m_per_s,
            use_gpu=use_gpu,
            verbose=False,
            save_preview=True,
            progress_callback=_capture_line,
        )
    finally:
        stop_event.set()
        timer.join(timeout=2)
        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)
        _sys.stderr.write(
            f"\r[recon] ✓ done in {mins}m {secs}s"
            + " " * 60 + "\n"
        )
        _sys.stderr.flush()

    ws.recon = result
    ws.venc_m_per_s = venc_m_per_s
    ws.voxel_size_mm = (voxel_size_mm_dz, voxel_size_mm_dy, voxel_size_mm_dx)
    ws.suggested_seeds = None
    info = {
        "status": "ok",
        "source": "fresh_recon",
        "method": method,
        "n_iterations": n_iterations,
        "shape_ZYXT": list(result["xHat"].shape),
        "output_path": result.get("output_path"),
        "preview_paths": result.get("preview_paths", []),
        "elapsed_wall_s": round(time.time() - t0, 1),
        "matlab_elapsed_minutes": result.get("meta", {}).get("elapsed_minutes"),
        "venc_m_per_s": venc_m_per_s,
        "voxel_size_mm": list(ws.voxel_size_mm),
    }
    _persist_recon(ws, info)
    return info


# How often to refresh the elapsed-time status line during a MATLAB recon.
_RECON_PROGRESS_INTERVAL_S = 3.0


# ----- load_as4df (Stanford physical phantom + STL ground truth) ----------

def _tool_load_as4df(ws: Workspace,
                     dataset_root: str | None = None,
                     model: str = "m_c1",
                     n_frames: int = 50,
                     load_stl_mask: bool = True,
                     mask_name: str = "aorta_stl") -> dict:
    """
    Load a Stanford AS4DF (Aortic Stiffness 4D Flow) acquisition into the
    workspace. AS4DF is a 3D-printed compliant aorta phantom imaged with
    standard 4D flow MRI sequences, with STL meshes giving the ground-truth
    vessel geometry — this lets us evaluate segmentation against truth.

    Host overrides:
        When the runner has set env vars MEDICT_AS4DF_ROOT / MEDICT_AS4DF_MODEL
        / MEDICT_AS4DF_N_FRAMES / MEDICT_AS4DF_LOAD_STL, those win over
        whatever the LLM passed as arguments. This is the anti-hallucination
        anchor for small LLMs that tend to fabricate paths (e.g.
        /mnt/g/medict_tmp/...) even when the goal text says otherwise.
    """
    import os
    env_root  = os.environ.get("MEDICT_AS4DF_ROOT")
    env_model = os.environ.get("MEDICT_AS4DF_MODEL")
    env_n     = os.environ.get("MEDICT_AS4DF_N_FRAMES")
    env_stl   = os.environ.get("MEDICT_AS4DF_LOAD_STL")
    if env_root:  dataset_root  = env_root
    if env_model: model         = env_model
    if env_n:     n_frames      = int(env_n)
    if env_stl is not None:
        load_stl_mask = env_stl.strip().lower() in ("1", "true", "yes", "on")
    if not dataset_root:
        return {"error": "dataset_root not supplied and no MEDICT_AS4DF_ROOT env var",
                "tool":  "load_as4df"}

    from skills.dicom_loader import load_as4df, voxelize_stl_to_mask

    recon = load_as4df(dataset_root, model=model,
                       n_frames=n_frames, use_corrected=True)
    ws.recon            = {k: recon[k] for k in ("xHat", "thetaX", "thetaY", "thetaZ")}
    ws.venc_m_per_s     = recon["venc_m_per_s"]
    ws.voxel_size_mm    = recon["voxel_size_mm"]
    ws.dt_seconds       = recon["dt_seconds"]
    ws.suggested_seeds  = None

    info: dict = {
        "status":        "loaded",
        "source":        "AS4DF",
        "dataset_root":  str(dataset_root),    # ★ recorded for downstream UI panels
        "model":         model,
        "n_frames":      n_frames,
        "load_stl_mask": load_stl_mask,
        "shape_ZYXT":    list(recon["thetaX"].shape),
        "venc_m_per_s":  recon["venc_m_per_s"],
        "voxel_size_mm": list(recon["voxel_size_mm"]),
    }

    if load_stl_mask:
        stl_path = Path(dataset_root) / "stl"
        candidates = sorted(stl_path.glob("*.stl"),
                            key=lambda p: p.stat().st_size, reverse=True)
        if candidates:
            mask = voxelize_stl_to_mask(
                candidates[0],
                shape_ZYX=recon["thetaX"].shape[:3],
                voxel_size_mm=recon["voxel_size_mm"],
                origin_mm=recon.get("origin_mm"),       # DICOM-grid alignment
            )
            ws.masks[mask_name] = mask
            info["mask_name"]     = mask_name
            info["mask_voxels"]   = int(mask.sum())
            info["stl_used"]      = candidates[0].name
            if ws.store is not None:
                ws.store.write_mask(mask_name, mask=mask, meta={
                    "source":    "AS4DF STL voxelization",
                    "stl_file":  candidates[0].name,
                    "n_voxels":  int(mask.sum()),
                })
        else:
            info["mask_warning"]  = "no STL files found under <root>/stl/"

    _persist_recon(ws, info)
    return info


# ----- load_phantom (good-case demo) --------------------------------------

def _tool_load_phantom(ws: Workspace,
                       mask_name: str = "aorta_phantom",
                       venc_m_per_s: float = 1.5) -> dict:
    """
    Load a synthetic curved-tapered "aorta" phantom into the workspace.

    The phantom is a single straight tube along Z with a smooth radius taper
    (7 → 5 voxels), an incompressible analytically-constructed velocity field
    (∇·v = 0), and pulsatile cardiac variation. Its ground-truth mask is
    placed in workspace.masks[mask_name].

    Use this for the "good case" demo: bypasses MATLAB reconstruction and
    PC-MRA segmentation, exercising only the Physics Verifier and Hemodynamic
    Analyzer specialists with a known-clean controlled input.
    """
    from skills.eval_inject._phantom_v2 import curved_tapered_phantom

    p = curved_tapered_phantom(venc_m_per_s=venc_m_per_s, pulsatile=True)
    ws.recon = {
        "thetaX": p["thetaX"],
        "thetaY": p["thetaY"],
        "thetaZ": p["thetaZ"],
        # Placeholder anatomy — verifier/analyzer don't read this, but tools
        # that call require_recon() check the dict shape.
        "xHat":   np.ones(p["thetaX"].shape, dtype=np.complex64),
    }
    ws.venc_m_per_s    = p["venc_m_per_s"]
    ws.voxel_size_mm   = p["voxel_size_mm"]
    ws.dt_seconds      = p["dt_seconds"]
    ws.suggested_seeds = None
    ws.masks[mask_name] = p["mask"]
    info = {
        "status":         "phantom_loaded",
        "source":         "synthetic_phantom",
        "mask_name":      mask_name,
        "shape_ZYXT":     list(p["thetaX"].shape),
        "venc_m_per_s":   p["venc_m_per_s"],
        "voxel_size_mm":  list(p["voxel_size_mm"]),
        "n_mask_voxels":  int(p["mask"].sum()),
        "geometry":       p.get("geometry", "tapered_incompressible"),
        "note":           "ground-truth mask already in workspace; no segmentation needed",
    }
    _persist_recon(ws, info)
    if ws.store is not None:
        ws.store.write_mask(mask_name, mask=p["mask"], meta={
            "source":   "synthetic phantom analytic mask",
            "geometry": p.get("geometry", "tapered_incompressible"),
            "n_voxels": int(p["mask"].sum()),
        })
    return info


# ----- suggest_seeds -------------------------------------------------------

def _tool_suggest_seeds(ws: Workspace, n_candidates: int = 5,
                        percentile: float = 90.0,
                        closing_iter: int = 1) -> dict:
    r = ws.require_recon()
    candidates = _suggest_seed_points(
        r["thetaX"], r["thetaY"], r["thetaZ"], r["xHat"],
        ws.venc_m_per_s,
        n_candidates=n_candidates,
        percentile=percentile,
        closing_iter=closing_iter,
    )
    # Sanitize: drop the bbox for prompt brevity, keep what the LLM needs to decide
    summary = []
    for i, c in enumerate(candidates):
        z0, y0, x0, z1, y1, x1 = c["bbox"]
        summary.append({
            "index":       i,
            "seed_zyx":    [int(v) for v in c["seed"]],
            "size_voxels": int(c["size"]),
            "mean_pcmra":  round(float(c["mean_pcmra"]), 4),
            "bbox_dims":   [int(z1 - z0), int(y1 - y0), int(x1 - x0)],
        })
    ws.suggested_seeds = candidates
    if ws.store is not None:
        ws.store.write_seed_suggestions(summary)
    return {"candidates": summary, "n_returned": len(summary)}


# ----- segment_from_seed ---------------------------------------------------

def _tool_segment_from_seed(ws: Workspace, seed_z: int, seed_y: int, seed_x: int,
                            mask_name: str,
                            percentile: float = 90.0,
                            closing_iter: int = 1) -> dict:
    r = ws.require_recon()
    mask = _segment_from_seed(
        r["thetaX"], r["thetaY"], r["thetaZ"], r["xHat"],
        ws.venc_m_per_s,
        seed_point=(seed_z, seed_y, seed_x),
        percentile=percentile,
        closing_iter=closing_iter,
    )
    if not mask.any():
        return {
            "status": "empty_mask",
            "seed_zyx": [seed_z, seed_y, seed_x],
            "warning": "segmentation returned no voxels — seed may be in background; "
                       "lower percentile or pick a different seed.",
        }
    ws.masks[mask_name] = mask

    # Quick stats so the LLM can decide whether to keep this mask
    speed = np.sqrt(
        (r["thetaX"] * ws.venc_m_per_s / np.pi) ** 2
        + (r["thetaY"] * ws.venc_m_per_s / np.pi) ** 2
        + (r["thetaZ"] * ws.venc_m_per_s / np.pi) ** 2
    )
    peak = float(speed[mask].max())

    info = {
        "status":            "ok",
        "mask_name":         mask_name,
        "seed_zyx":          [seed_z, seed_y, seed_x],
        "size_voxels":       int(mask.sum()),
        "peak_speed_m_per_s": round(peak, 3),
        "percentile":        percentile,
        "closing_iter":      closing_iter,
    }
    if ws.store is not None:
        ws.store.write_mask(mask_name, mask=mask, meta={
            "source":             "segment_from_seed",
            "seed_zyx":           [seed_z, seed_y, seed_x],
            "percentile":         percentile,
            "closing_iter":       closing_iter,
            "n_voxels":           int(mask.sum()),
            "peak_speed_m_per_s": round(peak, 3),
        })
    return info


# ----- verify --------------------------------------------------------------

def _tool_verify(ws: Workspace, mask_name: str) -> dict:
    r = ws.require_recon()
    mask = ws.require_mask(mask_name)
    verdict = _verify(
        r["thetaX"], r["thetaY"], r["thetaZ"],
        venc_m_per_s=ws.venc_m_per_s,
        mask=mask,
        voxel_size_mm=ws.voxel_size_mm,
    )
    ws.verdicts[mask_name] = verdict
    safe = to_json_safe(verdict)
    if ws.store is not None:
        ws.store.write_verdict(mask_name, {"mask_name": mask_name, **safe})
    return {"mask_name": mask_name, **safe}


# ----- analyze -------------------------------------------------------------

def _tool_analyze(ws: Workspace, mask_name: str,
                  n_cross_sections: int = 5) -> dict:
    r = ws.require_recon()
    mask = ws.require_mask(mask_name)
    report = _analyze(
        r["thetaX"], r["thetaY"], r["thetaZ"],
        venc_m_per_s=ws.venc_m_per_s,
        mask=mask,
        voxel_size_mm=ws.voxel_size_mm,
        dt_seconds=ws.dt_seconds,
        n_cross_sections=n_cross_sections,
    )
    ws.analyses[mask_name] = report
    safe = to_json_safe(report)
    if ws.store is not None:
        ws.store.write_analysis(mask_name, {"mask_name": mask_name, **safe})
    return {"mask_name": mask_name, **safe}


# ----- crop_mask (single-segment extraction) ------------------------------

_AXIS_TO_INDEX = {"Z": 0, "Y": 1, "X": 2}


def _tool_crop_mask(ws: Workspace,
                    source_mask: str,
                    target_mask: str,
                    axis: str = "Z",
                    start_frac: float = 0.0,
                    end_frac: float = 1.0) -> dict:
    """Extract a sub-region of `source_mask` along one coordinate axis and
    save it as a new mask `target_mask`. Used to isolate a single tubular
    segment from a branched mask (e.g. cropping a whole-aorta mask down to
    just the descending portion) so the single-vessel net_flux verifier
    can apply meaningfully.
    """
    src = ws.require_mask(source_mask)
    if axis not in _AXIS_TO_INDEX:
        return {"error": f"axis must be one of {list(_AXIS_TO_INDEX)}, got {axis!r}",
                "tool":  "crop_mask"}
    if not (0.0 <= start_frac < end_frac <= 1.0):
        return {"error": "require 0.0 <= start_frac < end_frac <= 1.0",
                "tool":  "crop_mask"}
    if target_mask in ws.masks:
        return {"error": f"target_mask {target_mask!r} already exists; pick a new name",
                "tool":  "crop_mask"}

    ax = _AXIS_TO_INDEX[axis]
    n = src.shape[ax]
    i0 = int(round(n * start_frac))
    i1 = int(round(n * end_frac))
    cropped = np.zeros_like(src)
    sl = [slice(None)] * src.ndim
    sl[ax] = slice(i0, i1)
    cropped[tuple(sl)] = src[tuple(sl)]
    ws.masks[target_mask] = cropped
    info = {
        "status":          "ok",
        "source_mask":     source_mask,
        "target_mask":     target_mask,
        "axis":            axis,
        "start_frac":      start_frac,
        "end_frac":        end_frac,
        "axis_index_range":[i0, i1],
        "n_voxels_before": int(src.sum()),
        "n_voxels_after":  int(cropped.sum()),
    }
    if ws.store is not None:
        ws.store.write_mask(target_mask, mask=cropped, meta={
            "source":      f"crop of {source_mask}",
            "axis":        axis,
            "start_frac":  start_frac,
            "end_frac":    end_frac,
            "n_voxels":    int(cropped.sum()),
        })
    return info


# ----- read_file (scoped) --------------------------------------------------

# Hard cap on how many characters any one read_file call may return to the
# LLM. Mask metadata + verdict JSON are typically a few KB; this guards
# against an accidental request for a binary or oversized file getting
# rendered into the prompt.
_READ_FILE_MAX_CHARS = 32_000


def _tool_read_file(ws: Workspace, path: str) -> dict:
    """Read a session-relative artifact file. Permission is checked against
    READ_SCOPES for ws.current_role; path traversal is blocked by the store."""
    if ws.store is None:
        return {"error": "no session store attached to workspace", "tool": "read_file"}
    role = ws.current_role or "<unknown>"
    if not is_path_allowed(role, path):
        return {
            "error": f"role {role!r} is not allowed to read {path!r}",
            "tool":  "read_file",
            "role":  role,
            "allowed_prefixes": _allowed_prefixes_for(role),
        }
    try:
        text = ws.store.read_relative(path)
    except FileNotFoundError as e:
        return {"error": str(e), "tool": "read_file", "path": path}
    except PermissionError as e:
        return {"error": str(e), "tool": "read_file", "path": path}
    truncated = len(text) > _READ_FILE_MAX_CHARS
    return {
        "path":       path,
        "n_chars":    len(text),
        "truncated":  truncated,
        "content":    text[:_READ_FILE_MAX_CHARS],
    }


def _tool_list_dir(ws: Workspace, path: str = "") -> dict:
    """List names in one session sub-directory (one level deep). Same scope
    rules as read_file."""
    if ws.store is None:
        return {"error": "no session store attached to workspace", "tool": "list_dir"}
    role = ws.current_role or "<unknown>"
    # For directory listing, allow the empty / root case for any role that
    # has at least one prefix (i.e. can read something) — they need a way to
    # discover what subfolders exist.
    if path and not is_path_allowed(role, path.rstrip("/") + "/"):
        return {
            "error": f"role {role!r} is not allowed to list {path!r}",
            "tool":  "list_dir",
            "role":  role,
            "allowed_prefixes": _allowed_prefixes_for(role),
        }
    return {"path": path, "entries": ws.store.list_relative(path)}


def _allowed_prefixes_for(role: str) -> list[str]:
    from .session_store import READ_SCOPES
    return list(READ_SCOPES.get(role, []))


# ============================================================================
# Tool registry
# ============================================================================

TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="load_reconstruction",
        description=(
            "Load a previously-saved Stage 2a reconstruction (.mat) into the workspace. "
            "Use this for fast iteration when a reconstruction already exists. "
            "Replaces any currently-loaded reconstruction."
        ),
        parameters={
            "type": "object",
            "properties": {
                "mat_path":          {"type": "string", "description": "Filesystem path to a Stage 2a output .mat"},
                "venc_m_per_s":      {"type": "number", "default": 1.5, "minimum": 0.1, "maximum": 10.0},
                "voxel_size_mm_dz":  {"type": "number", "default": 2.0, "minimum": 0.1, "maximum": 10.0},
                "voxel_size_mm_dy":  {"type": "number", "default": 2.0, "minimum": 0.1, "maximum": 10.0},
                "voxel_size_mm_dx":  {"type": "number", "default": 2.0, "minimum": 0.1, "maximum": 10.0},
            },
            "required": ["mat_path"],
        },
        function=_tool_load_reconstruction,
    ),
    ToolSpec(
        name="load_as4df",
        description=(
            "Load a Stanford AS4DF (Aortic Stiffness 4D Flow) acquisition: a real "
            "MRI scan of a 3D-printed compliant aortic phantom, with optional STL "
            "ground-truth mask. Real PC-MRA contrast + real noise, but the vessel "
            "geometry is known exactly — ideal for evaluating segmentation. Set "
            "load_stl_mask=True to also load the ground-truth mask into the "
            "workspace (skips segmentation entirely; useful for Verifier/Hemodynamic "
            "demos). Set False to leave segmentation to the agent."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dataset_root":  {"type": "string",
                                  "description": "Path to AS4DF root, containing 'dicoms/' and 'stl/' folders"},
                "model":         {"type": "string", "default": "m_c1",
                                  "enum": ["m_c1", "m_c2", "m_r"]},
                "n_frames":      {"type": "integer", "default": 50,
                                  "enum": [16, 25, 50]},
                "load_stl_mask": {"type": "boolean", "default": True,
                                  "description": "Voxelize the STL into a ground-truth vessel mask"},
                "mask_name":     {"type": "string", "default": "aorta_stl"},
            },
            "required": [],   # all params resolvable via env vars set by host
        },
        function=_tool_load_as4df,
    ),
    ToolSpec(
        name="load_phantom",
        description=(
            "Load a synthetic curved-tapered aorta phantom into the workspace. "
            "Use this for controlled 'good case' demos: the phantom is a single "
            "tapered tube along Z with an analytically-constructed incompressible "
            "velocity field and pulsatile cardiac variation. Ground-truth mask is "
            "placed in workspace.masks[mask_name] automatically — NO segmentation "
            "step is required afterward. Designed to pass all 4 Physics Verifier checks."
        ),
        parameters={
            "type": "object",
            "properties": {
                "mask_name":    {"type": "string", "default": "aorta_phantom",
                                 "description": "Name to store the ground-truth mask under"},
                "venc_m_per_s": {"type": "number", "default": 1.5,
                                 "minimum": 0.1, "maximum": 10.0},
            },
            "required": [],
        },
        function=_tool_load_phantom,
    ),
    ToolSpec(
        name="reconstruct",
        description=(
            "Run a fresh CS or CORe reconstruction from raw k-space (slow — 1-12 minutes on RTX 5090). "
            "Use 5 iterations for a smoke test, 50 for full quality. "
            "Only call this when no reconstruction is loaded or when the current one is too noisy."
        ),
        parameters={
            "type": "object",
            "properties": {
                "kspace_path":      {"type": "string", "description": "Path to input k-space .mat"},
                "method":           {"type": "string", "enum": ["cs", "core"], "default": "cs"},
                "n_iterations":     {"type": "integer", "default": 50, "minimum": 1, "maximum": 200},
                "venc_m_per_s":     {"type": "number", "default": 1.5, "minimum": 0.1, "maximum": 10.0},
                "voxel_size_mm_dz": {"type": "number", "default": 2.0, "minimum": 0.1, "maximum": 10.0},
                "voxel_size_mm_dy": {"type": "number", "default": 2.0, "minimum": 0.1, "maximum": 10.0},
                "voxel_size_mm_dx": {"type": "number", "default": 2.0, "minimum": 0.1, "maximum": 10.0},
                "use_gpu":          {"type": "boolean", "default": True},
            },
            "required": ["kspace_path"],
        },
        function=_tool_reconstruct,
    ),
    ToolSpec(
        name="suggest_seeds",
        description=(
            "Return the top-N candidate vessel regions from the current reconstruction, "
            "ranked by brightness × size. Each candidate has a seed coordinate, size in "
            "voxels, mean PC-MRA, and bbox dimensions. Call once after loading recon; the "
            "result is cached. Use this to pick which seed_zyx to pass to segment_from_seed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "n_candidates": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                "percentile":   {"type": "number", "default": 90.0, "minimum": 50.0, "maximum": 99.9},
                "closing_iter": {"type": "integer", "default": 1, "minimum": 0, "maximum": 5},
            },
            "required": [],
        },
        function=_tool_suggest_seeds,
    ),
    ToolSpec(
        name="segment_from_seed",
        description=(
            "Region-grow a vessel mask from a (z, y, x) seed voxel. Stores it under "
            "mask_name. Lower percentile = bigger mask. Returns mask size + peak speed; "
            "if peak < ~0.5 m/s the seed probably hit a noise region — try another."
        ),
        parameters={
            "type": "object",
            "properties": {
                "seed_z":       {"type": "integer", "minimum": 0},
                "seed_y":       {"type": "integer", "minimum": 0},
                "seed_x":       {"type": "integer", "minimum": 0},
                "mask_name":    {"type": "string", "description": "Name to store this mask under (LLM-chosen, e.g. 'aorta_v1')"},
                "percentile":   {"type": "number", "default": 90.0, "minimum": 50.0, "maximum": 99.9},
                "closing_iter": {"type": "integer", "default": 1, "minimum": 0, "maximum": 5},
            },
            "required": ["seed_z", "seed_y", "seed_x", "mask_name"],
        },
        function=_tool_segment_from_seed,
    ),
    ToolSpec(
        name="verify",
        description=(
            "Run the 4 Physics Verifier checks (divergence, net flux, peak velocity, "
            "phase unwrap) on a named mask. Returns a verdict (pass/warn/fail) with "
            "per-check numerical results. Use this to decide whether a mask is good "
            "enough to feed to analyze."
        ),
        parameters={
            "type": "object",
            "properties": {
                "mask_name": {"type": "string"},
            },
            "required": ["mask_name"],
        },
        function=_tool_verify,
    ),
    ToolSpec(
        name="analyze",
        description=(
            "Run hemodynamic analysis (volumetric flow Q(t), stroke volume, peak flow, "
            "peak velocity) on a named mask. Returns per-cross-section results and a "
            "summary. Only call after verify() if you want validated results."
        ),
        parameters={
            "type": "object",
            "properties": {
                "mask_name":        {"type": "string"},
                "n_cross_sections": {"type": "integer", "default": 5, "minimum": 2, "maximum": 20},
            },
            "required": ["mask_name"],
        },
        function=_tool_analyze,
    ),
    ToolSpec(
        name="crop_mask",
        description=(
            "Extract a sub-region of an existing mask along one coordinate "
            "axis, saving the result as a new mask. Use this when a mask "
            "covers a branched vessel (whole aorta = ascending + arch + "
            "descending + supra-aortic branches) and the single-vessel "
            "net_flux verifier returned `status: skip` for that reason. "
            "Cropping to a fractional range along Z (or Y, X) of the most "
            "tubular segment lets net_flux check a region where its "
            "single-vessel assumption holds.\n\n"
            "Typical aortic-arch isolation: axis='Z', start_frac=0.0, "
            "end_frac=0.5 keeps the bottom half of the volume (descending "
            "aorta) and drops the arch + ascending + branches."
        ),
        parameters={
            "type": "object",
            "properties": {
                "source_mask": {"type": "string",
                                "description": "Name of the existing mask to crop."},
                "target_mask": {"type": "string",
                                "description": "Name to save the cropped sub-mask under."},
                "axis":        {"type": "string", "default": "Z",
                                "enum": ["Z", "Y", "X"]},
                "start_frac":  {"type": "number", "default": 0.0,
                                "minimum": 0.0, "maximum": 1.0},
                "end_frac":    {"type": "number", "default": 1.0,
                                "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["source_mask", "target_mask"],
        },
        function=_tool_crop_mask,
    ),
    ToolSpec(
        name="read_file",
        description=(
            "Read a session artifact (JSON / text file) by its session-relative "
            "path. Use this to read your own tool outputs before writing a "
            "report so the numbers you cite are grounded in actual saved data, "
            "NOT remembered from prior reasoning. Permission is enforced per "
            "specialist role; reading outside your allowed prefixes returns an "
            "error listing which prefixes you may access."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "Session-relative path, e.g. "
                                        "'verification/verify_aorta_v1.json'."},
            },
            "required": ["path"],
        },
        function=_tool_read_file,
    ),
    ToolSpec(
        name="list_dir",
        description=(
            "List file names directly under a session sub-directory. Use this "
            "to discover what artifacts exist (e.g. which mask names have been "
            "produced) before deciding which file to read_file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "",
                         "description": "Session-relative dir, e.g. "
                                        "'segmentation/masks/'. Empty for root."},
            },
            "required": [],
        },
        function=_tool_list_dir,
    ),
]

TOOLS_BY_NAME: dict[str, ToolSpec] = {t.name: t for t in TOOLS}


# ============================================================================
# Dispatcher
# ============================================================================

def call_tool(workspace: Workspace, name: str, args: dict) -> dict:
    """Validate ``args`` against the named tool's schema, dispatch, sanitize.

    Returns a JSON-safe dict. On error returns ``{"error": <msg>, "error_type": ...}``
    so the LLM can react rather than crashing the loop.
    """
    spec = TOOLS_BY_NAME.get(name)
    if spec is None:
        return {
            "error": f"unknown tool {name!r}",
            "error_type": "UnknownTool",
            "known_tools": sorted(TOOLS_BY_NAME),
        }

    try:
        validated = _validate_args(args, spec.parameters)
        result = spec.function(workspace, **validated)
        return to_json_safe(result)
    except ToolError as e:
        return {"error": str(e), "error_type": "ToolError", "tool": name}
    except Exception as e:
        return {"error": str(e), "error_type": type(e).__name__, "tool": name}


# ============================================================================
# Prompt formatting — for stuffing into the agent's system prompt
# ============================================================================

def tools_prompt_block(tools: list[ToolSpec] | None = None) -> str:
    """Render the tool list as a readable block for the LLM system prompt."""
    tools = tools if tools is not None else TOOLS
    out = ["## Tools available\n"]
    for t in tools:
        out.append(f"### {t.name}\n{t.description.strip()}\n")
        out.append("Parameters (JSON Schema):\n```json\n"
                   + json.dumps(t.parameters, indent=2)
                   + "\n```\n")
    return "\n".join(out)
