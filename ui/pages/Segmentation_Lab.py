"""[V2 stub] Manual ROI / SAM2 segmentation playground."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="MEDICT — Segmentation Lab", page_icon="🔬", layout="wide")
st.title("Segmentation Lab")
st.caption("V2 — coming soon")

st.info(
    "Future: upload a recon, draw a bbox / ROI on a 2D slice, "
    "let SAM2 propagate to 3D, push the mask into the agent workspace."
)
