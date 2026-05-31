"""Centerline-aware net-flux check (V2 Tier 2 #6).

The axis-aligned `check_net_flux` in `_checks.py` assumes one mask = one
straight-ish tube aligned with a coordinate axis. For curved vessels
(aortic arch, tortuous) or branched masks the assumption fails: the dominant
axis flux measurement counts opposing flow components against each other and
reports false non-conservation.

This module computes flux at cross-sections perpendicular to the LOCAL
centerline tangent, not to a global coordinate axis — the way real 4D-flow
clinical software (CAAS, Circle CVI) does. For an incompressible flow in
a vessel of varying curvature, flux through any properly-oriented cross-
section should be conserved.

Pipeline:
    1. Skeletonize the binary mask (scikit-image)
    2. Extract a single backbone path (longest geodesic) — branches are
       discarded; this is a single-vessel verifier for V1+1.
    3. Resample to N evenly-spaced cross-sections
    4. At each section, compute local tangent via central differences
    5. Volume-integrate (v · tangent) over a thin slab perpendicular to t
       → flux through the section
    6. Compute max deviation across sections (continuity test)

Out-of-scope for this V2 prototype:
    - Branched vessels (Y / T) — picks the longest backbone and ignores branches
    - Non-isotropic voxels — uses voxel-volume/slab-thickness conversion that
      assumes the slab perpendicular-to-tangent area ≈ voxel cross-section area
"""
from __future__ import annotations

import numpy as np
from collections import deque


# ─────────────────────────────────────────────────────────────────────
# Skeleton + centerline extraction
# ─────────────────────────────────────────────────────────────────────

def _skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """3D skeletonization via scikit-image (works on bool array, returns bool)."""
    from skimage.morphology import skeletonize
    return skeletonize(mask.astype(bool))


def _build_skeleton_graph(skel: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Return (coords[N,3], adjacency: list of neighbor index lists).
    26-connectivity in 3D."""
    coords = np.argwhere(skel)                    # (N, 3) int
    if len(coords) == 0:
        return coords, []
    # Build dict: coord tuple → index for O(N) neighbor lookup
    idx_of = {tuple(c): i for i, c in enumerate(coords)}
    neighbors: list[list[int]] = [[] for _ in coords]
    offsets = [(dz, dy, dx)
               for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dz, dy, dx) != (0, 0, 0)]
    for i, (z, y, x) in enumerate(coords):
        for dz, dy, dx in offsets:
            key = (int(z + dz), int(y + dy), int(x + dx))
            j = idx_of.get(key)
            if j is not None:
                neighbors[i].append(j)
    return coords, neighbors


def _farthest_node(start: int, adj: list[list[int]]) -> tuple[int, dict[int, int]]:
    """BFS — returns (farthest node index, parent dict for path reconstruction)."""
    dist = {start: 0}
    parent: dict[int, int] = {start: -1}
    q = deque([start])
    farthest = start
    while q:
        u = q.popleft()
        if dist[u] > dist[farthest]:
            farthest = u
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
    return farthest, parent


def _longest_path(coords: np.ndarray, adj: list[list[int]]) -> list[int]:
    """Double-BFS: longest geodesic in a tree (approximate for general graph)."""
    if len(coords) == 0:
        return []
    # BFS from arbitrary node → reach farthest leaf u
    u, _ = _farthest_node(0, adj)
    # BFS from u → reach farthest leaf v; backtrack u → v via parents
    v, parent = _farthest_node(u, adj)
    path = []
    cur = v
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


def count_skeleton_branches(mask: np.ndarray | None) -> int:
    """
    Number of branch points in the mask's 3D skeleton (nodes with degree >= 3).

    Used by the verifier checks to detect when a mask covers a branched
    vessel (e.g. aortic arch with supra-aortic branches). Both `net_flux`
    and `net_flux_centerline` assume a single tubular vessel; when this
    helper reports >= 1 branch the checks return `status: skip` rather
    than producing a misleading FAIL on a structurally valid multi-branch
    mask.

    Returns 0 if mask is None / empty / has no skeleton.
    """
    if mask is None or not mask.any():
        return 0
    skel = _skeletonize_mask(mask)
    if skel.sum() < 4:
        return 0
    _, adj = _build_skeleton_graph(skel)
    return sum(1 for neigh in adj if len(neigh) >= 3)


def extract_centerline(mask: np.ndarray,
                        n_sections: int = 12) -> np.ndarray | None:
    """
    Backbone centerline of `mask`, evenly resampled to `n_sections` points.

    Returns (N, 3) float array in (z, y, x) voxel coordinates, or None
    if the mask has no usable skeleton (too small / disconnected).
    """
    skel = _skeletonize_mask(mask)
    if skel.sum() < 4:
        return None
    coords, adj = _build_skeleton_graph(skel)
    path = _longest_path(coords, adj)
    if len(path) < 4:
        return None
    pts = coords[path].astype(np.float32)                         # (P, 3)

    # Resample uniformly by arc-length to n_sections points
    deltas = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(deltas)))                # cumulative arc length
    total = s[-1]
    if total < 1.0:
        return None
    targets = np.linspace(0.0, total, n_sections)
    resampled = np.empty((n_sections, 3), dtype=np.float32)
    for k, t in enumerate(targets):
        i = int(np.searchsorted(s, t, side="right") - 1)
        i = max(0, min(i, len(pts) - 2))
        u = (t - s[i]) / max(deltas[i], 1e-6)
        resampled[k] = (1 - u) * pts[i] + u * pts[i + 1]
    return resampled


def _compute_tangents(centerline: np.ndarray,
                       voxel_size_mm: tuple[float, float, float]) -> np.ndarray:
    """
    Unit tangent vectors at each centerline point (in physical space,
    so tangents are anisotropy-corrected).
    Returns (N, 3) array of unit vectors in (z, y, x) ordering.
    """
    vox = np.asarray(voxel_size_mm, dtype=np.float32)
    pts_mm = centerline * vox                                     # to physical mm
    tangents = np.zeros_like(pts_mm)
    tangents[1:-1] = pts_mm[2:] - pts_mm[:-2]
    tangents[0]    = pts_mm[1]  - pts_mm[0]
    tangents[-1]   = pts_mm[-1] - pts_mm[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return tangents / norms


# ─────────────────────────────────────────────────────────────────────
# Flux at a single cross-section
# ─────────────────────────────────────────────────────────────────────

def _orthonormal_basis(tangent_unit_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to `tangent_unit_mm`,
    in physical (mm) coordinates with (z, y, x) ordering."""
    t = np.asarray(tangent_unit_mm, dtype=np.float32)
    # Use the world axis least parallel to t to seed the basis
    helper = np.eye(3, dtype=np.float32)[int(np.argmin(np.abs(t)))]
    u1 = np.cross(t, helper); u1 /= np.linalg.norm(u1)
    u2 = np.cross(t, u1)                                          # already unit
    return u1, u2


def _flux_through_section(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                           mask: np.ndarray,
                           voxel_size_mm: tuple[float, float, float],
                           point_vox: np.ndarray,
                           tangent_unit_mm: np.ndarray,
                           sample_radius_mm: float,
                           n_samples_per_side: int = 40) -> float:
    """
    Q [mL/s] through the cross-section perpendicular to `tangent_unit_mm` at
    `point_vox`. Computed by trilinear-interpolating the velocity field and
    mask on a regular 2D grid sampled in the cross-section plane.

    Robust to centerline / voxel-grid alignment (no slab-membership thresholds).
    `tangent_unit_mm` is a unit vector in PHYSICAL space (mm), (z, y, x) ordering.
    """
    from scipy.ndimage import map_coordinates

    dz, dy, dx = voxel_size_mm                                    # mm
    u1, u2 = _orthonormal_basis(tangent_unit_mm)

    # Sample grid in the cross-section plane (mm-coordinates)
    grid = np.linspace(-sample_radius_mm, sample_radius_mm, n_samples_per_side)
    U, V = np.meshgrid(grid, grid, indexing="ij")

    p_mm = np.array([point_vox[0] * dz, point_vox[1] * dy, point_vox[2] * dx],
                    dtype=np.float32)
    # World-coordinate of each sample = p + U*u1 + V*u2
    samples_mm = (p_mm[None, None, :]
                  + U[:, :, None] * u1[None, None, :]
                  + V[:, :, None] * u2[None, None, :])
    # Convert to voxel coords for map_coordinates
    samples_vox = samples_mm / np.array([dz, dy, dx], dtype=np.float32)
    coords = samples_vox.reshape(-1, 3).T                         # (3, M*M)

    # Time-average fields (steady-state assumption)
    vx_t = vx.mean(axis=-1) if vx.ndim == 4 else vx
    vy_t = vy.mean(axis=-1) if vy.ndim == 4 else vy
    vz_t = vz.mean(axis=-1) if vz.ndim == 4 else vz

    vz_s = map_coordinates(vz_t.astype(np.float32), coords, order=1,
                            mode="constant", cval=0.0).reshape(U.shape)
    vy_s = map_coordinates(vy_t.astype(np.float32), coords, order=1,
                            mode="constant", cval=0.0).reshape(U.shape)
    vx_s = map_coordinates(vx_t.astype(np.float32), coords, order=1,
                            mode="constant", cval=0.0).reshape(U.shape)
    mask_s = map_coordinates(mask.astype(np.float32), coords, order=1,
                              mode="constant", cval=0.0).reshape(U.shape)

    # Project velocity onto tangent → m/s; mask hard-threshold at 0.5
    v_dot_t = vz_s * tangent_unit_mm[0] + vy_s * tangent_unit_mm[1] + vx_s * tangent_unit_mm[2]
    in_mask = mask_s > 0.5

    du_mm = float(grid[1] - grid[0])
    sample_area_m2 = (du_mm * du_mm) * 1e-6
    return float(np.sum(v_dot_t[in_mask]) * sample_area_m2) * 1e6  # → mL/s


# ─────────────────────────────────────────────────────────────────────
# Public check
# ─────────────────────────────────────────────────────────────────────

def check_net_flux_centerline(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    mask: np.ndarray | None,
    voxel_size_mm: tuple[float, float, float],
    n_sections: int = 12,
) -> dict:
    """
    Centerline-aware flux conservation. Works on curved / non-axis-aligned
    single-vessel masks where the standard axis-aligned `check_net_flux`
    breaks down.

    Output mirrors `check_net_flux` for drop-in interoperability.
    """
    if mask is None or not mask.any():
        return {
            "status": "skip",
            "reason": "no mask supplied",
            "requires_segmentation": True,
        }

    # Branched-mask scope guard. The longest-path centerline ignores side
    # branches, so on a Y / T / aortic-arch-with-supra-aortic-branches mask
    # the flux estimate would silently undercount and the verdict could
    # FALSELY FAIL on a structurally correct mask. Mark SKIP instead.
    n_branches = count_skeleton_branches(mask)
    if n_branches >= 1:
        return {
            "status": "skip",
            "reason": (f"mask skeleton has {n_branches} branch point(s); "
                       "out of single-vessel verifier scope. Use crop_mask "
                       "to isolate one segment, or accept that V1 flux "
                       "conservation does not apply across branches."),
            "n_skeleton_branches": int(n_branches),
            "n_mask_voxels":       int(mask.sum()),
            "requires_segmentation": True,
        }

    centerline = extract_centerline(mask, n_sections=n_sections)
    if centerline is None:
        return {
            "status": "skip",
            "reason": "mask skeleton too small or disconnected",
            "n_mask_voxels": int(mask.sum()),
            "requires_segmentation": True,
        }

    tangents = _compute_tangents(centerline, voxel_size_mm)

    # Sample-plane geometry — anchored on the vessel's equivalent radius so
    # density adapts to the mask, not to the grid extent.
    centerline_length_mm = float(np.linalg.norm(
        np.diff(centerline, axis=0) * np.asarray(voxel_size_mm),
        axis=1).sum())
    mask_volume_mm3 = float(mask.sum() * np.prod(voxel_size_mm))
    mean_equiv_radius_mm = float(np.sqrt(
        max(mask_volume_mm3 / max(centerline_length_mm, 1.0), 1.0) / np.pi))
    sample_radius_mm = 2.5 * mean_equiv_radius_mm
    sample_step_mm   = 0.7 * float(min(voxel_size_mm))
    n_samples = max(40, int(round(2 * sample_radius_mm / sample_step_mm)))

    # Skip the first and last 10% of sections — endpoint tangents are unreliable
    # (one-sided differences at termini).
    inner = slice(max(1, n_sections // 10), n_sections - max(1, n_sections // 10))
    flux_values: list[float] = []
    for p, t in zip(centerline[inner], tangents[inner]):
        flux_values.append(round(
            _flux_through_section(vx, vy, vz, mask, voxel_size_mm,
                                  p, t, sample_radius_mm,
                                  n_samples_per_side=n_samples),
            3,
        ))

    arr = np.array(flux_values)
    ref = float(np.abs(arr).mean()) or 1.0
    max_dev_pct = float(np.max(np.abs(arr - arr.mean())) / ref * 100)

    WARN_PCT, FAIL_PCT = 10.0, 25.0
    status = "fail" if max_dev_pct > FAIL_PCT else "warn" if max_dev_pct > WARN_PCT else "pass"

    return {
        "status":             status,
        "flux_mL_per_s":      flux_values,
        "max_deviation_pct":  round(max_dev_pct, 2),
        "threshold_warn_pct": WARN_PCT,
        "threshold_fail_pct": FAIL_PCT,
        "n_sections":         len(flux_values),
        "sample_radius_mm":   round(sample_radius_mm, 2),
        "sample_step_mm":     round(sample_step_mm, 2),
        "n_samples_per_side": n_samples,
        "centerline_length_mm": round(centerline_length_mm, 2),
        "mean_equiv_radius_mm": round(mean_equiv_radius_mm, 2),
        "method":             "centerline-tangent-perpendicular",
        "requires_segmentation": True,
    }
