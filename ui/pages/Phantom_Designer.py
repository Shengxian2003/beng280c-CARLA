"""[V2 stub] Tune phantom parameters and run verifier interactively."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="MEDICT — Phantom Designer", page_icon="🧪", layout="wide")
st.title("Phantom Designer")
st.caption("V2 — coming soon")

st.info(
    "Future: drag sliders for tube radius / taper / pulsatile amplitude, "
    "see verifier verdict update in real time."
)
