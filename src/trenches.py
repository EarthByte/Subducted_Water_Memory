"""Plate-boundary sample points at a reconstruction time.

Lifted verbatim from the reconstruction stage of an earlier project so that this
repository does not import from it. The tessellation threshold, the returned
columns and the swallow-and-return-empty behaviour on a time the model cannot
resolve are all unchanged, so distances computed here match the earlier run.
"""
import numpy as np

TESSELLATION_RADIANS = 0.01


def boundary_points(recon, t):
    """Subduction and ridge sample points at time t.

    Returns lon, lat and convergence or spreading rate in cm/yr for each.
    """
    out = {}
    try:
        sz = recon.tessellate_subduction_zones(
            float(t), tessellation_threshold_radians=TESSELLATION_RADIANS,
            ignore_warnings=True)
        out['subduction'] = (sz[:, 0], sz[:, 1], sz[:, 2])
    except Exception:
        out['subduction'] = (np.array([]), np.array([]), np.array([]))
    try:
        mor = recon.tessellate_mid_ocean_ridges(
            float(t), tessellation_threshold_radians=TESSELLATION_RADIANS,
            ignore_warnings=True)
        out['ridge'] = (mor[:, 0], mor[:, 1], mor[:, 2])
    except Exception:
        out['ridge'] = (np.array([]), np.array([]), np.array([]))
    return out
