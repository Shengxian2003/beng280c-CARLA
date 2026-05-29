# MEDICT UI

Streamlit-based control panel for the MEDICT pipeline. Designed for V2 to absorb
new tools (SAM2, branch-aware verifier, phantom tuning) without restructuring.

## Install

```bash
# In the medict conda env:
pip install streamlit
```

## Launch

From the project root:

```bash
streamlit run ~/projects/medict/ui/app.py

```

Streamlit will print a URL (usually `http://localhost:8501`). Open it in a browser.

## Pages

| Page | Purpose |
|---|---|
| `app.py` (Demo Runner) | Pick mode + LLM, run the pipeline, see live results |
| `Audit Viewer` | Browse any historical audit log under `logs/` |
| `Phantom Designer` | (V2) Tune phantom params and watch verifier respond |
| `Segmentation Lab` | (V2) Manual ROI / SAM2 segmentation playground |

## Folder layout

```
ui/
├── app.py                          # Main entry (Demo Runner)
├── pages/                          # Auto-discovered by Streamlit
│   ├── Audit_Viewer.py
│   ├── Phantom_Designer.py    # V2 stub
│   └── Segmentation_Lab.py    # V2 stub
├── _widgets/                       # Reusable panels (leading _ = not a page)
│   ├── audit_timeline.py
│   ├── verifier_panel.py
│   └── hemodynamic_panel.py
└── _utils/
    └── runner.py                   # Wraps existing demos/*.py via subprocess
```

## Adding a new page in V2

Drop a file in `ui/pages/` named `N_emoji_Name.py`. Streamlit will pick it up
automatically and add it to the sidebar nav. Reuse the `_widgets/` panels and
`_utils/runner.py` for consistency.

## How runs work

`ui/_utils/runner.py` invokes the existing `demos/good_case_demo.py` or
`demos/single_window_demo.py` as a subprocess. Output streams live to the UI;
when the run finishes, the audit log (`logs/ui_*.jsonl`) is parsed back into
verdicts + hemodynamic results for the structured display.

This means: any improvement to the underlying demo scripts shows up in the UI
with zero UI changes.
