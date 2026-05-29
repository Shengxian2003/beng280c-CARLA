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
from typing import Any, Callable

import numpy as np

from skills.reconstruction import reconstruct as _reconstruct
from skills.segmentation import (
    suggest_seed_points as _suggest_seed_points,
    segment_from_seed as _segment_from_seed,
)
from skills.physics_verifier import verify as _verify
from skills.hemodynamic import analyze as _analyze


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
    return {
        "status": "loaded",
        "source_path": mat_path,
        "shape_ZYXT": list(ws.recon["xHat"].shape),
        "venc_m_per_s": venc_m_per_s,
        "voxel_size_mm": list(ws.voxel_size_mm),
    }


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
    return {
        "status": "ok",
        "method": method,
        "n_iterations": n_iterations,
        "shape_ZYXT": list(result["xHat"].shape),
        "output_path": result.get("output_path"),
        "preview_paths": result.get("preview_paths", []),
        "elapsed_wall_s": round(time.time() - t0, 1),
        "matlab_elapsed_minutes": result.get("meta", {}).get("elapsed_minutes"),
    }


# How often to refresh the elapsed-time status line during a MATLAB recon.
_RECON_PROGRESS_INTERVAL_S = 3.0


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
    return {
        "status":         "phantom_loaded",
        "mask_name":      mask_name,
        "shape_ZYXT":     list(p["thetaX"].shape),
        "venc_m_per_s":   p["venc_m_per_s"],
        "voxel_size_mm":  list(p["voxel_size_mm"]),
        "n_mask_voxels":  int(p["mask"].sum()),
        "geometry":       p.get("geometry", "tapered_incompressible"),
        "note":           "ground-truth mask already in workspace; no segmentation needed",
    }


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

    return {
        "status":            "ok",
        "mask_name":         mask_name,
        "seed_zyx":          [seed_z, seed_y, seed_x],
        "size_voxels":       int(mask.sum()),
        "peak_speed_m_per_s": round(peak, 3),
        "percentile":        percentile,
        "closing_iter":      closing_iter,
    }


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
    return {"mask_name": mask_name, **to_json_safe(verdict)}


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
    return {"mask_name": mask_name, **to_json_safe(report)}


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
