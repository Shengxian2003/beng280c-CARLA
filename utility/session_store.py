"""Filesystem-backed session store for agent I/O grounding.

Why this exists
---------------
LLM specialists were observed fabricating numerical results — e.g. the
verifier specialist reported "divergence 37.35 s-1" while all five of its
underlying ``verify`` tool calls had returned errors ("mask not found").
Pure prompt-based defenses were not sufficient.

The session store persists every tool's input metadata, output JSON, and
output binary blob to disk under a per-session directory. Combined with:

  * a scoped ``read_file`` tool that lets each specialist read only its
    declared input/output folders,
  * a numeric-grounding check at done-emission time that requires every
    number a specialist quotes in its report to appear verbatim in a tool
    output the specialist read,

this makes hallucinated metrics structurally rejectable rather than only
discouraged by prompt language.

Directory layout
----------------
.. code-block:: text

    medict/logs/runs/<session_id>/
      input/
        manifest.json
      recon/
        summary.json
        recon.npz           # xHat + thetaX/Y/Z; LLM does not read this
      segmentation/
        seed_suggestions.json
        masks/
          <name>.meta.json  # voxels, peak_speed, seed, params
          <name>.npy        # binary mask
      verification/
        verify_<mask>.json
      hemodynamic/
        analyze_<mask>.json
      coordinator/
        plan.json
        delegations.jsonl

Each tool dual-writes: it updates the in-memory ``Workspace`` (so existing
tools stay fast) AND writes the artifact to its session folder (so the
filesystem is a complete grounded snapshot the LLM can re-read).

This module is intentionally dependency-free — only stdlib + numpy. Pure
infrastructure; no LLM coupling.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ----- Directory names (kept as constants so callers don't fork strings) -----

DIR_INPUT          = "input"
DIR_RECON          = "recon"
DIR_SEGMENTATION   = "segmentation"
DIR_MASKS          = "segmentation/masks"
DIR_VERIFICATION   = "verification"
DIR_HEMODYNAMIC    = "hemodynamic"
DIR_COORDINATOR    = "coordinator"


@dataclass
class SessionStore:
    """Manages on-disk artifacts for one pipeline session.

    A SessionStore is created once per run (when the AuditLog is opened) and
    handed into Workspace. Every tool then calls ``store.write_*`` methods
    in addition to mutating the workspace.

    ``root`` is the parent directory; the session directory itself is
    ``root / session_id``. Created lazily on first write.
    """

    session_id: str
    root: Path                                # typically <project>/logs/runs/
    _created: set[Path] = field(default_factory=set)

    # ----- Path helpers ----------------------------------------------------

    @property
    def session_dir(self) -> Path:
        return self.root / self.session_id

    def _ensure(self, relative: str) -> Path:
        """Create (idempotent) and return one of the session sub-directories."""
        path = self.session_dir / relative
        if path not in self._created:
            path.mkdir(parents=True, exist_ok=True)
            self._created.add(path)
        return path

    # ----- Writers used by tools ------------------------------------------

    def write_input_manifest(self, manifest: dict) -> Path:
        out = self._ensure(DIR_INPUT) / "manifest.json"
        _write_json(out, manifest)
        return out

    def write_recon(self, *, summary: dict, arrays: dict[str, np.ndarray] | None = None) -> Path:
        """Write recon summary JSON (and optionally a recon.npz blob).

        ``summary`` is the LLM-readable metadata: shape, venc, voxel, source.
        ``arrays`` is keyword-name → ndarray, dumped as a single .npz so
        tools downstream (segmentation, verifier) can re-load if needed.
        """
        d = self._ensure(DIR_RECON)
        summary_path = d / "summary.json"
        _write_json(summary_path, summary)
        if arrays:
            np.savez_compressed(d / "recon.npz", **arrays)
        return summary_path

    def write_seed_suggestions(self, suggestions: list[dict]) -> Path:
        d = self._ensure(DIR_SEGMENTATION)
        path = d / "seed_suggestions.json"
        _write_json(path, {"seeds": suggestions})
        return path

    def write_mask(self, name: str, *, mask: np.ndarray, meta: dict) -> tuple[Path, Path]:
        """Write a segmentation mask: meta JSON + binary .npy. Returns (meta_path, blob_path)."""
        d = self._ensure(DIR_MASKS)
        meta_path = d / f"{name}.meta.json"
        blob_path = d / f"{name}.npy"
        # Save mask first so meta can record its true voxel count
        np.save(blob_path, mask.astype(bool))
        full_meta = {**meta, "name": name, "shape_ZYX": list(mask.shape),
                     "n_voxels_actual": int(mask.sum()),
                     "blob_path": str(blob_path.relative_to(self.session_dir))}
        _write_json(meta_path, full_meta)
        return meta_path, blob_path

    def write_verdict(self, mask_name: str, verdict: dict) -> Path:
        d = self._ensure(DIR_VERIFICATION)
        path = d / f"verify_{mask_name}.json"
        _write_json(path, verdict)
        return path

    def write_analysis(self, mask_name: str, analysis: dict) -> Path:
        d = self._ensure(DIR_HEMODYNAMIC)
        path = d / f"analyze_{mask_name}.json"
        _write_json(path, analysis)
        return path

    def write_plan(self, plan: dict) -> Path:
        d = self._ensure(DIR_COORDINATOR)
        path = d / "plan.json"
        _write_json(path, plan)
        return path

    def append_delegation(self, entry: dict) -> Path:
        d = self._ensure(DIR_COORDINATOR)
        path = d / "delegations.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=_jsonify) + "\n")
        return path

    # ----- Reader used by the scoped read_file tool ----------------------

    def read_relative(self, relative: str) -> str:
        """Return file contents (UTF-8 text) for a path under the session dir.

        Does NOT validate scope — that's the caller's job (the read_file
        tool enforces per-specialist prefixes). This method only protects
        against path traversal.
        """
        target = (self.session_dir / relative).resolve()
        if not str(target).startswith(str(self.session_dir.resolve())):
            raise PermissionError(f"path escapes session dir: {relative!r}")
        if not target.exists():
            raise FileNotFoundError(f"no such file in session: {relative!r}")
        return target.read_text(encoding="utf-8")

    def list_relative(self, relative: str) -> list[str]:
        """Return file names under one session sub-directory (one level deep)."""
        target = (self.session_dir / relative).resolve()
        if not str(target).startswith(str(self.session_dir.resolve())):
            raise PermissionError(f"path escapes session dir: {relative!r}")
        if not target.exists():
            return []
        return sorted(p.name for p in target.iterdir())


# ============================================================================
# JSON helpers — numpy-safe, preserves bool/int/float exactly
# ============================================================================

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, default=_jsonify, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")


def _jsonify(value: Any) -> Any:
    """Numpy → builtin coercion for json.dumps."""
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON-serializable: {type(value).__name__}")


__all__ = [
    "SessionStore",
    "DIR_INPUT", "DIR_RECON", "DIR_SEGMENTATION", "DIR_MASKS",
    "DIR_VERIFICATION", "DIR_HEMODYNAMIC", "DIR_COORDINATOR",
    "READ_SCOPES",
    "is_path_allowed",
]


# ============================================================================
# Per-specialist read scopes — used by the scoped read_file tool to decide
# which session-relative paths each specialist is allowed to read.
# ============================================================================
#
# Each entry maps a specialist role name to a list of path prefixes. A read
# is allowed if the requested path starts with ANY of the prefixes (treated
# as forward-slash prefix strings, so "recon/" matches "recon/summary.json"
# but not "reconstruction/").
#
# The "*" wildcard means full read access — used for the Coordinator which
# needs to see all specialist outputs to make routing decisions.
# ============================================================================

READ_SCOPES: dict[str, list[str]] = {
    "reconstruction": [
        # Recon specialist needs to know what input it was handed.
        f"{DIR_INPUT}/",
        # And to confirm its own output landed correctly.
        f"{DIR_RECON}/",
    ],
    "segmentation": [
        f"{DIR_INPUT}/",
        f"{DIR_RECON}/summary.json",
        # Its own outputs (seed suggestions + every mask's meta JSON).
        f"{DIR_SEGMENTATION}/",
    ],
    "verifier": [
        f"{DIR_RECON}/summary.json",
        f"{DIR_MASKS}/",
        # Crucially: the verifier must re-read its OWN verdict JSON before
        # writing its done report. This is the Stage-3 grounding anchor.
        f"{DIR_VERIFICATION}/",
    ],
    "hemodynamic": [
        f"{DIR_RECON}/summary.json",
        f"{DIR_MASKS}/",
        f"{DIR_VERIFICATION}/",
        f"{DIR_HEMODYNAMIC}/",
    ],
    "coordinator": ["*"],
    # Plan agents have no execution-phase reads.
    "planner":      [f"{DIR_INPUT}/"],
    "plan_critic":  [f"{DIR_INPUT}/", f"{DIR_COORDINATOR}/plan.json"],
}


def is_path_allowed(role: str, relative_path: str) -> bool:
    """True if `role` may read `relative_path` (session-relative)."""
    scopes = READ_SCOPES.get(role, [])
    if "*" in scopes:
        return True
    rp = relative_path.lstrip("/")
    return any(rp.startswith(prefix) for prefix in scopes)
