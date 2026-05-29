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


@dataclass(frozen=True)
class InputProfile:
    """Everything the pipeline needs to know about ONE input mode."""

    mode:          InputMode
    cli_flag:      str                # value passed via --input-mode
    ui_label:      str                # shown in the UI sidebar dropdown

    # Recon specialist
    recon_tools:   tuple[str, ...]    # EXACTLY the tools the LLM sees
    recon_prompt_suffix: str = ""     # appended to base RECONSTRUCTION_PROMPT

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


# ============================================================================
# Registry — single source of truth
# ============================================================================

_REAL_SCAN_PROFILE = InputProfile(
    mode              = InputMode.REAL_SCAN,
    cli_flag          = "real_scan",
    ui_label          = "Real 4D flow scan (.mat reconstruction)",
    recon_tools       = ("load_reconstruction", "reconstruct"),
    recon_prompt_suffix = "",
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
    seg_passthrough     = True,
    expected_mask_name  = "aorta_phantom",
    seg_prompt_suffix   = _PHANTOM_SEG_SUFFIX,
    goal_template       = _phantom_goal,
)


REGISTRY: dict[InputMode, InputProfile] = {
    InputMode.REAL_SCAN: _REAL_SCAN_PROFILE,
    InputMode.PHANTOM:   _PHANTOM_PROFILE,
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


def with_no_fresh_recon(profile: InputProfile) -> InputProfile:
    """
    Return a copy of `profile` with the `reconstruct` tool removed and a
    demo-mode suffix appended to the recon prompt.

    Only meaningful for REAL_SCAN (PHANTOM doesn't have `reconstruct` anyway).
    For other modes this is a no-op.
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
]
