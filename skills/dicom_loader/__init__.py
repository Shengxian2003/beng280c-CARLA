"""
AS4DF DICOM loader.

Reads Stanford CMR Group's AS4DF (Aortic Stiffness 4D Flow) dataset:
  - 3 model variants (m_c1 compliant, m_c2 compliant, m_r rigid)
  - 3 temporal resolutions (16 / 25 / 50 frames per cycle)
  - 4 DICOM series per acquisition: magnitude + 3 phase (vx, vy, vz)
  - STL meshes giving the *ground-truth aortic wall* — voxelizable into a mask

Output matches our workspace.recon dict format so the same Verifier and
Hemodynamic specialists can be used downstream without changes.
"""
from __future__ import annotations

from ._loader import (
    AS4DF_MODELS, AS4DF_TEMPORAL_FRAMES,
    discover_as4df_series, load_as4df, voxelize_stl_to_mask,
)

__all__ = [
    "AS4DF_MODELS", "AS4DF_TEMPORAL_FRAMES",
    "discover_as4df_series", "load_as4df", "voxelize_stl_to_mask",
]
