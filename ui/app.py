"""
MEDICT Streamlit UI — main entry.

Run from project root:
    streamlit run ui/app.py

App-mode philosophy: user provides a scan path; system analyzes it and reports
honestly. Synthetic phantom is offered as a built-in option for demos / testing.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from utility.input_modes import REGISTRY as INPUT_PROFILES
from ui._utils.runner import InputType, RunConfig, kill_orphan_run, run_demo
from ui._widgets.agent_reports import render_agent_reports
from ui._widgets.audit_timeline import render_timeline
from ui._widgets.hemodynamic_panel import render_hemodynamic
from ui._widgets.image_panel import render_images
from ui._widgets.pipeline_panel import PipelineState, render_pipeline
from ui._widgets.summary_panel import render_summary
from ui._widgets.verifier_panel import render_verifier

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MEDICT",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Drop stale session state if any schema changed (e.g. after a code update)
_stale = st.session_state.get("last_run")
if _stale is not None and (
    not hasattr(_stale, "pipeline_state")            # RunResult schema bumped
    or not hasattr(_stale, "aborted")                # added on stop-button change
    or not hasattr(_stale.config, "input_type")      # RunConfig schema bumped
    or not hasattr(_stale.config, "llm_model")       # model picker added
    or (_stale.pipeline_state is not None
        and "summarizer" not in _stale.pipeline_state.agents)   # summarizer added
):
    del st.session_state["last_run"]


# ─────────────────────────────────────────────────────────────────────
# Sidebar — what to analyze + how
# ─────────────────────────────────────────────────────────────────────

st.sidebar.title("🧬 MEDICT")
st.sidebar.caption("Multi-Agent 4D Flow MRI Pipeline")

st.sidebar.header("Input")
# Sidebar options are derived from the input-mode registry — adding a new
# profile in agents/input_modes.py is enough to make it appear here.
INPUT_LABELS = {
    InputType(p.cli_flag): p.ui_label for p in INPUT_PROFILES.values()
}
input_label = st.sidebar.selectbox(
    "What to analyze",
    options=list(INPUT_LABELS.keys()),
    format_func=lambda k: INPUT_LABELS[k],
    index=0,
)
input_type = input_label

scan_path: str | None = None
venc        = 1.5
voxel_mm    = 2.0

if input_type == InputType.REAL_SCAN:
    # Quick presets that resolve to known paths on this system.
    presets = {
        "(type custom path below)":               "",
        "OSU-MR 50-iter CS recon (full quality)": "/mnt/g/medict_tmp/recon_cs_50iter.mat",
        "OSU-MR 5-iter CS recon (smoke quality)": "/mnt/g/medict_tmp/recon_cs_5iter.mat",
    }
    preset_choice = st.sidebar.selectbox("Quick presets", list(presets.keys()), index=1)
    default_path  = presets[preset_choice]
    scan_path = st.sidebar.text_input(
        "Reconstruction path (.mat)",
        value=default_path,
        help="Path to a Stage-2a-format reconstruction output (.mat) "
             "with outputs.xHat / thetaX / thetaY / thetaZ.",
    )

    with st.sidebar.expander("Acquisition parameters", expanded=False):
        venc      = st.number_input("VENC (m/s)", value=1.5, min_value=0.1, max_value=10.0, step=0.1)
        voxel_mm  = st.number_input("Voxel size (mm, isotropic)", value=2.0, min_value=0.1, max_value=10.0, step=0.1)

st.sidebar.header("LLM backend")
BACKEND_LABELS = {
    "mock":   "mock (scripted, ~5 s — wiring tests)",
    "ollama": "ollama (local Qwen via GPU)",
}
llm_label = st.sidebar.selectbox(
    "Reasoning engine",
    options=list(BACKEND_LABELS.keys()),
    format_func=lambda k: BACKEND_LABELS[k],
    index=0,
)
llm_backend = llm_label

OLLAMA_MODELS = {
    "qwen2.5:7b-instruct":  "Qwen 2.5 7B Instruct — fast, recommended for iteration",
    "qwen3.6":              "Qwen 3.6 35B — heavy, full reasoning, slow",
}
if llm_backend == "ollama":
    llm_model = st.sidebar.selectbox(
        "Ollama model",
        options=list(OLLAMA_MODELS.keys()),
        index=0,
        format_func=lambda m: f"{m}  —  {OLLAMA_MODELS[m].split('—', 1)[1].strip()}",
        help="Model must be already pulled via `ollama pull <name>`.",
    )
else:
    llm_model = "qwen3.6"   # placeholder, ignored when mock

with st.sidebar.expander("Advanced options"):
    max_plan_revisions = st.slider(
        "Max plan revisions",
        min_value=0, max_value=3, value=0,
        help="How many times the Planner can rewrite its plan after Plan Critic feedback.",
    )
    max_delegations = st.slider(
        "Max coordinator delegations",
        min_value=5, max_value=20, value=12,
        help="Safety cap on Coordinator → Specialist delegations.",
    )

st.sidebar.divider()
run_disabled = (
    input_type == InputType.REAL_SCAN and not (scan_path or "").strip()
)
run_button = st.sidebar.button(
    "▶ Analyze",
    type="primary",
    use_container_width=True,
    disabled=run_disabled,
)

# Cancel button — only effective during a running pipeline. Sets a flag the
# runner's main loop checks every 0.4s.
def _request_cancel():
    st.session_state["_cancel_run"] = True

st.sidebar.button(
    "⏹ Stop (discard log)",
    on_click=_request_cancel,
    type="secondary",
    use_container_width=True,
    help="Cooperatively cancel the running pipeline within ~1 second. "
         "The partial audit log is deleted.",
)

# Hard kill any orphaned subprocess from a previous browser session
if st.session_state.get("_running_pid"):
    if st.sidebar.button("🚨 KILL SWITCH",
                          type="secondary",
                          use_container_width=True,
                          help="Hard-kills any background pipeline subprocess "
                               "(e.g. one orphaned when the browser closed) and "
                               "discards its log."):
        if kill_orphan_run():
            st.sidebar.success("☠️ Process killed. Log discarded.")
        else:
            st.sidebar.info("No background process to kill.")


# ─────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────

st.title("MEDICT — 4D Flow Hemodynamic Analysis")

# Two-column layout: main content on left (3/4), pipeline panel on right (1/4)
main_col, pipeline_col = st.columns([3, 1])

# Pipeline placeholder always present, so it renders both pre-run + during-run
pipeline_placeholder = pipeline_col.empty()

# Render an idle pipeline panel by default
_idle_state = PipelineState()
render_pipeline(_idle_state, pipeline_placeholder)

def _check_backend_reachable(backend: str) -> tuple[bool, str]:
    """Quick ping so we fail fast with a friendly error instead of a long
    stack trace from inside the subprocess."""
    import requests
    if backend == "mock":
        return True, ""
    if backend == "ollama":
        try:
            requests.get("http://localhost:11434/", timeout=2)
            return True, ""
        except Exception as e:
            return False, (
                f"Cannot reach Ollama at http://localhost:11434: {type(e).__name__}. "
                "Start it with `ollama serve` (or check it's running)."
            )
    return True, ""

if run_button:
    ok, err = _check_backend_reachable(llm_backend)
    if not ok:
        st.sidebar.error(err)
        st.stop()

    cfg = RunConfig(
        input_type         = input_type,
        scan_path          = scan_path,
        venc_m_per_s       = venc,
        voxel_size_mm      = voxel_mm,
        llm_backend        = llm_backend,
        llm_model          = llm_model,
        max_plan_revisions = max_plan_revisions,
        max_delegations    = max_delegations,
    )
    with main_col:
        st.subheader("Pipeline output")
        st.caption("Live stdout (last 60 lines)")
        stdout_placeholder = st.empty()
    st.session_state["last_run"] = run_demo(
        cfg,
        pipeline_placeholder=pipeline_placeholder,
        stdout_placeholder=stdout_placeholder,
    )

last = st.session_state.get("last_run")

# After a run, re-render pipeline with final state
if last is not None and last.pipeline_state is not None:
    render_pipeline(last.pipeline_state, pipeline_placeholder)

# All main-area content lives inside main_col so the pipeline column stays put
with main_col:
    if last is None:
        # ── Welcome / instructions ─────────────────────────────────────
        st.info(
            "Pick an input (a real scan path or the built-in phantom) in the sidebar, "
            "choose an LLM backend, and click **▶ Analyze**. Results appear here."
        )
        with st.expander("How the pipeline behaves", expanded=True):
            st.markdown(
                """
                The pipeline runs the same way regardless of input. **What changes
                is what each agent finds**:

                - **Reconstruction operator** loads the velocity field (or the phantom).
                - **Segmentation operator** isolates a vessel mask.
                - **Physics Verifier** runs four deterministic checks
                  (divergence, net flux, peak velocity, phase unwrap).
                  **The verifier decides whether the system trusts the data.**
                - **Hemodynamic Analyzer** only runs if the verifier is satisfied.
                  It produces flow Q(t), stroke volume, and peak velocity.

                Each specialist runs under an **energy budget**: a small number of
                tool calls plus thinking rounds, visible to the LLM. When the
                budget runs low, the specialist must wrap up with a best-effort
                report rather than reasoning indefinitely. Watch the pipeline
                column on the right for live agent status and remaining budgets.
                """
            )
    else:
        # ── Run result ─────────────────────────────────────────────────
        if last.exit_code == 0:
            st.success(
                f"Analysis completed in {last.elapsed_s:.1f} s · "
                f"status: **{last.session_status}** · "
                f"{last.n_delegations} delegations"
            )
        else:
            st.error(f"Run failed (exit code {last.exit_code}) in {last.elapsed_s:.1f} s")

        # st.tabs() does NOT persist its selection across script reruns
        # (every slider drag → full rerun → tab reset to first). Use a radio
        # with a session_state key so the active "tab" survives reruns.
        TAB_NAMES = [
            "📊 Results", "🖼 Images", "📝 Agent reports",
            "📜 Audit timeline", "🖥 Raw output", "💾 Files",
        ]
        active_tab = st.radio(
            "view_tab", TAB_NAMES,
            horizontal=True,
            key="active_results_tab",      # stored in session_state
            label_visibility="collapsed",
        )
        st.divider()

        if active_tab == TAB_NAMES[0]:        # Results
            col_v, col_h = st.columns(2)
            with col_v:
                render_verifier(last.verdicts)
            with col_h:
                render_hemodynamic(last.analyses)
            st.divider()
            render_summary(last.audit_path)

        elif active_tab == TAB_NAMES[1]:      # Images
            render_images(last.audit_path)

        elif active_tab == TAB_NAMES[2]:      # Agent reports
            render_agent_reports(last.audit_path)

        elif active_tab == TAB_NAMES[3]:      # Audit timeline
            render_timeline(last.audit_path)

        elif active_tab == TAB_NAMES[4]:      # Raw output
            st.code(last.stdout_tail, language="text")

        elif active_tab == TAB_NAMES[5]:      # Files
            st.markdown(f"**Audit log:** `{last.audit_path}`")
            if last.audit_path.exists():
                with open(last.audit_path, "rb") as f:
                    st.download_button(
                        "Download audit log (JSONL)",
                        data=f.read(),
                        file_name=last.audit_path.name,
                        mime="application/jsonl",
                    )
            with st.expander("Run configuration"):
                cfg = last.config
                st.json({
                    "input_type":         cfg.input_type.value,
                    "scan_path":          cfg.scan_path,
                    "venc_m_per_s":       cfg.venc_m_per_s,
                    "voxel_size_mm":      cfg.voxel_size_mm,
                    "llm_backend":        cfg.llm_backend,
                    "llm_model":          cfg.llm_model,
                    "max_plan_revisions": cfg.max_plan_revisions,
                    "max_delegations":    cfg.max_delegations,
                })
