"""Input-mode profiles — one source of truth for each kind of run.

Why this file exists:
    Pre-refactor, the recon specialist's tool list was controlled by two
    boolean flags (`phantom_mode`, `demo_mode`) and `build_default_specialists`
    chose tools and prompts by combining them. That worked when there were
    only two loaders, but it scaled badly: adding a third loader (AS4DF)
    exposed both `load_phantom` and `load_as4df` to the LLM at once and small
    models picked the wrong one, poisoning the run upstream.

    Each input mode now owns the full set of decisions that vary with the
    input: which loader(s) the recon specialist may call, what suffix the
    recon prompt gets, whether segmentation passes through a pre-loaded
    mask, and how the natural-language goal is phrased. UI / runner / demo
    script / specialist factory all read this same registry.

    Adding a future input mode (re-adding AS4DF, adding nnU-Net masks, etc.)
    is a single new entry here — no edits in specialist.py, no new flags.

Key invariant:
    `profile.recon_tools` is the EXACT and ONLY list of tools the recon
    specialist sees. The LLM cannot call a tool that isn't on that list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


# ============================================================================
# Enum + dataclass
# ============================================================================

class InputMode(str, Enum):
    """The kinds of input the pipeline knows how to start from."""
    REAL_SCAN = "real_scan"
    PHANTOM   = "phantom"
    AS4DF     = "as4df"


@dataclass(frozen=True)
class InputProfile:
    """Everything the pipeline needs to know about ONE input mode."""

    mode:          InputMode
    cli_flag:      str                # value passed via --input-mode
    ui_label:      str                # shown in the UI sidebar dropdown

    # Recon specialist
    recon_tools:   tuple[str, ...]    # EXACTLY the tools the LLM sees
    recon_prompt_suffix: str = ""     # appended to base RECONSTRUCTION_PROMPT

    # True iff this input mode supports re-running the recon step with
    # different parameters (the only mode for which this is true is
    # REAL_SCAN with `reconstruct` in its tools — i.e. fresh MATLAB allowed).
    # When False, the Coordinator MUST NOT re-delegate to reconstruction
    # after the initial load — the load is one-shot. This addresses the
    # observed failure mode where the Coordinator re-delegated to the
    # reconstruction specialist 3 of 11 times on an AS4DF run, wasting
    # delegations on retries that could not change anything (DICOM data,
    # no iterations to tune).
    supports_reconstruction_retry: bool = True

    # Segmentation specialist
    seg_passthrough:     bool = False
    expected_mask_name:  Optional[str] = None
    seg_prompt_suffix:   str = ""     # appended to base SEGMENTATION_PROMPT

    # Goal-template builder. Receives a dict of runtime options (scan_path,
    # venc, voxel_mm, custom_goal, ...) and returns the user-facing goal
    # string. Defaults vary by mode and live with the profile, not in runner.
    goal_template: Callable[[dict], str] = field(default=lambda opts: "")


# ============================================================================
# Prompt suffix strings — co-located with the profile that uses them so
# loader-specific guidance never leaks into other modes' system prompts.
# ============================================================================

_PHANTOM_RECON_SUFFIX = """\

## PHANTOM MODE — important constraint
You are running on the built-in synthetic curved-tapered phantom (no MATLAB
reconstruction). Call `load_phantom` exactly once and emit done. The phantom
ships with its ground-truth mask, so the Segmentation specialist will pass
through without needing to call any segmentation tools.
"""

_PHANTOM_SEG_SUFFIX = """\

## ⚠ PASSTHROUGH MODE — STRICT CONSTRAINT ⚠
A ground-truth mask called 'aorta_phantom' is ALREADY in the workspace,
placed there by `load_phantom`. You MUST NOT call `suggest_seeds` or
`segment_from_seed` — those would create a redundant inferior mask and
waste your energy budget.

Your ONE AND ONLY action this turn:
  {"done": true, "report": "Passing through pre-loaded mask aorta_phantom.
   No segmentation needed; downstream specialists will use it by name."}
"""

# Real-scan demo-mode suffix (no fresh MATLAB recon allowed)
_REAL_SCAN_DEMO_SUFFIX = """\

## DEMO MODE — important constraint
You are running in demo mode. Fresh MATLAB reconstruction takes ~12 minutes
and is not available right now: the `reconstruct` tool is removed from your
toolset. Always use `load_reconstruction` with the existing 5-iter recon.
If the verifier later flags quality issues, do not request a re-run — let
the Hemodynamic Analyzer add appropriate caveats to its report.
"""

# AS4DF mode (Stanford 3D-printed aortic phantom + STL ground truth)
_AS4DF_RECON_SUFFIX = """\

## AS4DF MODE — important constraint
You are running on the Stanford AS4DF dataset (a 3D-printed compliant aortic
phantom imaged with real 4D flow MRI, shipped with an STL mesh of the printed
geometry). Call `load_as4df` exactly once with the dataset_root, model, and
n_frames provided in the task, then emit done. When `load_stl_mask=True` the
STL is voxelized onto the DICOM grid and placed in the workspace as a
ground-truth mask, so the Segmentation specialist will pass through.

PATH RULE: pass the `dataset_root` string from the task EXACTLY as given.
Do NOT prepend any prefix; the path is already absolute.
"""

_AS4DF_SEG_SUFFIX = """\

## ⚠ PASSTHROUGH MODE — STRICT CONSTRAINT ⚠
A ground-truth mask called 'aorta_stl' is ALREADY in the workspace, placed
there by `load_as4df` voxelizing the STL mesh onto the DICOM grid. You MUST
NOT call `suggest_seeds` or `segment_from_seed` — they would create a redundant
inferior mask and waste your energy budget.

Your ONE AND ONLY action this turn:
  {"done": true, "report": "Passing through pre-loaded mask aorta_stl.
   No segmentation needed; downstream specialists will use it by name."}
"""


# ============================================================================
# Goal-template builders
# ============================================================================

_REAL_SCAN_GOAL_TEMPLATE = (
    "Analyze the hemodynamics of the 4D flow MRI scan at {path}. "
    "VENC = {venc} m/s, voxel size {vox} mm isotropic. "
    "Pick a vessel, verify the physics, and report flow metrics. "
    "If verification fails, explain why and what should be done next."
)


def _real_scan_goal(opts: dict) -> str:
    if opts.get("custom_goal"):
        return opts["custom_goal"]
    return _REAL_SCAN_GOAL_TEMPLATE.format(
        path = opts.get("scan_path", "<missing scan_path>"),
        venc = opts.get("venc_m_per_s", 1.5),
        vox  = opts.get("voxel_size_mm", 2.0),
    )


def _phantom_goal(opts: dict) -> str:
    if opts.get("custom_goal"):
        return opts["custom_goal"]
    return (
        "Analyze the built-in synthetic curved-tapered aorta phantom "
        "(VENC=1.5 m/s, voxel 2mm). Load it via load_phantom — the "
        "ground-truth mask is placed in the workspace automatically. "
        "Skip segmentation, verify the physics, and report flow metrics."
    )


def _as4df_goal(opts: dict) -> str:
    if opts.get("custom_goal"):
        return opts["custom_goal"]
    load_stl_mask = bool(opts.get("load_stl_mask", True))
    tail = (
        "The STL ground-truth mask is loaded by the tool automatically. "
        "Skip segmentation, verify the physics on `aorta_stl`, then report flow metrics."
        if load_stl_mask else
        "Run segmentation to produce a vessel mask, then verify and report flow metrics."
    )
    # The dataset path, model, n_frames, and STL toggle are pre-configured
    # by the runner via environment variables — `load_as4df` reads them
    # directly. The LLM does NOT need to (and must not) pass these arguments;
    # just call the tool once with no args.
    return (
        f"Analyze the Stanford AS4DF dataset.\n\n"
        f"Call `load_as4df` exactly once with NO ARGUMENTS — every parameter "
        f"(dataset path, model, frame count, STL toggle) has already been "
        f"pre-configured by the host. Do NOT pass `dataset_root`, `model`, "
        f"`n_frames`, or `load_stl_mask` — the host overrides will reject "
        f"or replace anything you provide.\n\n"
        f"{tail}"
    )


# ============================================================================
# Registry — single source of truth
# ============================================================================

_REAL_SCAN_PROFILE = InputProfile(
    mode              = InputMode.REAL_SCAN,
    cli_flag          = "real_scan",
    ui_label          = "Real 4D flow scan (.mat reconstruction)",
    recon_tools       = ("load_reconstruction", "reconstruct"),
    recon_prompt_suffix = "",
    # Real scan with fresh recon CAN be retried with more iterations.
    supports_reconstruction_retry = True,
    seg_passthrough     = False,
    expected_mask_name  = None,
    seg_prompt_suffix   = "",
    goal_template       = _real_scan_goal,
)

_PHANTOM_PROFILE = InputProfile(
    mode              = InputMode.PHANTOM,
    cli_flag          = "phantom",
    ui_label          = "Synthetic phantom (built-in, for demos)",
    recon_tools       = ("load_phantom",),
    recon_prompt_suffix = _PHANTOM_RECON_SUFFIX,
    # load_phantom is deterministic and idempotent — there is no parameter
    # to tune, so a re-delegation can only waste budget.
    supports_reconstruction_retry = False,
    seg_passthrough     = True,
    expected_mask_name  = "aorta_phantom",
    seg_prompt_suffix   = _PHANTOM_SEG_SUFFIX,
    goal_template       = _phantom_goal,
)

_AS4DF_PROFILE = InputProfile(
    mode              = InputMode.AS4DF,
    cli_flag          = "as4df",
    ui_label          = "AS4DF DICOM (Stanford 3D-printed phantom + STL)",
    recon_tools       = ("load_as4df",),
    recon_prompt_suffix = _AS4DF_RECON_SUFFIX,
    # AS4DF data is loaded directly from DICOM — there is no reconstruction
    # to re-run with more iterations.
    supports_reconstruction_retry = False,
    seg_passthrough     = True,
    expected_mask_name  = "aorta_stl",
    seg_prompt_suffix   = _AS4DF_SEG_SUFFIX,
    goal_template       = _as4df_goal,
)


REGISTRY: dict[InputMode, InputProfile] = {
    InputMode.REAL_SCAN: _REAL_SCAN_PROFILE,
    InputMode.PHANTOM:   _PHANTOM_PROFILE,
    InputMode.AS4DF:     _AS4DF_PROFILE,
}


# ============================================================================
# Helpers
# ============================================================================

def get_profile(mode: InputMode | str) -> InputProfile:
    """Look up a profile by enum or by its CLI-flag string."""
    if isinstance(mode, InputMode):
        return REGISTRY[mode]
    for prof in REGISTRY.values():
        if prof.cli_flag == mode:
            return prof
    raise ValueError(
        f"unknown input mode {mode!r}; available: "
        f"{[p.cli_flag for p in REGISTRY.values()]}"
    )


def with_no_seg_passthrough(profile: InputProfile) -> InputProfile:
    """
    Return a copy of `profile` with segmentation passthrough disabled.

    Used when a mode that normally has a pre-loaded mask is run without it
    (e.g., AS4DF when the STL toggle is off). The segmentation specialist
    regains its tool allowlist and per-mode prompt suffix is stripped.
    """
    if not profile.seg_passthrough:
        return profile
    return InputProfile(
        mode               = profile.mode,
        cli_flag           = profile.cli_flag,
        ui_label           = profile.ui_label,
        recon_tools        = profile.recon_tools,
        recon_prompt_suffix = profile.recon_prompt_suffix,
        supports_reconstruction_retry = profile.supports_reconstruction_retry,
        seg_passthrough    = False,
        expected_mask_name = None,
        seg_prompt_suffix  = "",
        goal_template      = profile.goal_template,
    )


def with_no_fresh_recon(profile: InputProfile) -> InputProfile:
    """
    Return a copy of `profile` with the `reconstruct` tool removed and a
    demo-mode suffix appended to the recon prompt.

    Only meaningful for REAL_SCAN (PHANTOM doesn't have `reconstruct` anyway).
    For other modes this is a no-op. Also flips
    ``supports_reconstruction_retry`` to False — without `reconstruct` in
    the toolbox, a recon re-delegation cannot do anything new.
    """
    if "reconstruct" not in profile.recon_tools:
        return profile
    new_tools = tuple(t for t in profile.recon_tools if t != "reconstruct")
    return InputProfile(
        mode               = profile.mode,
        cli_flag           = profile.cli_flag,
        ui_label           = profile.ui_label,
        recon_tools        = new_tools,
        recon_prompt_suffix = profile.recon_prompt_suffix + _REAL_SCAN_DEMO_SUFFIX,
        supports_reconstruction_retry = False,
        seg_passthrough    = profile.seg_passthrough,
        expected_mask_name = profile.expected_mask_name,
        seg_prompt_suffix  = profile.seg_prompt_suffix,
        goal_template      = profile.goal_template,
    )


__all__ = [
    "InputMode",
    "InputProfile",
    "REGISTRY",
    "get_profile",
    "with_no_fresh_recon",
    "with_no_seg_passthrough",
]
