"""Utility functions for DMD windowing, data layout, and probe management.

Functions here are data-agnostic helpers used by both univar and multivar
runner scripts. They operate on numpy arrays and do not perform any I/O.

Functions
---------
cube_to_mat              — reshape (t, y, x) cube to (n_valid, t) matrix
mat_to_cube              — inverse of cube_to_mat
build_forecast_segments  — partition time series into train/forecast blocks
default_probe_points_xy  — auto-generate 5-point probe layout
snap_probe_to_valid      — snap a requested probe to the nearest valid grid point
build_probe_list         — convenience wrapper: validate and snap all probe points
run_method_over_segments — apply a forecast method across all segments
"""

import numpy as np


# =====================================================================
# CUBE / MATRIX CONVERSIONS
# =====================================================================

def cube_to_mat(cube, mask):
    """Reshape a (t, y, x) spatiotemporal cube to a (n_valid, t) matrix.

    Only spatially valid (unmasked) points are retained.
    DMD convention: columns = time snapshots, rows = spatial grid points.

    Parameters
    ----------
    cube : ndarray, shape (Nt, Ny, Nx)
    mask : bool ndarray, shape (Ny, Nx) — True at valid (non-land) points

    Returns
    -------
    mat : ndarray, shape (n_valid, Nt)
    """
    return cube.reshape(cube.shape[0], -1)[:, mask.ravel()].T


def mat_to_cube(mat, mask, Ny, Nx):
    """Reconstruct a (t, y, x) cube from a (n_valid, t) matrix.

    Invalid (land) points are filled with NaN.

    Parameters
    ----------
    mat  : ndarray, shape (n_valid, t)
    mask : bool ndarray, shape (Ny, Nx)
    Ny   : int
    Nx   : int

    Returns
    -------
    cube : ndarray, shape (t, Ny, Nx)  with NaN at masked points
    """
    t_dim = mat.shape[1]
    out = np.full((t_dim, Ny * Nx), np.nan)
    out[:, mask.ravel()] = mat.T
    return out.reshape(t_dim, Ny, Nx)


# =====================================================================
# WINDOWING
# =====================================================================

def build_forecast_segments(nt, use_alternating, n_train,
                            alt_train, alt_fc, alt_start):
    """Partition a time series into (train, forecast) window pairs.

    Parameters
    ----------
    nt              : int — total number of time steps
    use_alternating : bool — if True, cycle through record with fixed windows;
                             if False, use a single train block at the start
    n_train         : int — training block length (used when use_alternating=False)
    alt_train       : int — training block length per cycle (use_alternating=True)
    alt_fc          : int — forecast block length per cycle (use_alternating=True)
    alt_start       : int — starting time index for the first training block

    Returns
    -------
    segments : list of (tr0, tr1, fc0, fc1) tuples
        tr0, tr1 — slice [tr0:tr1] is the training block
        fc0, fc1 — slice [fc0:fc1] is the forecast block
    """
    segments = []
    if use_alternating:
        if alt_train < 2:
            raise ValueError('ALT_TRAIN_WINDOW must be at least 2.')
        if alt_fc < 1:
            raise ValueError('ALT_FORECAST_WINDOW must be at least 1.')
        if not (0 <= alt_start < nt):
            raise ValueError('ALT_START_INDEX must be in [0, Nt).')
        s = alt_start
        while s + alt_train < nt:
            tr0 = s
            tr1 = s + alt_train
            fc0 = tr1
            fc1 = min(fc0 + alt_fc, nt)
            if fc1 > fc0:
                segments.append((tr0, tr1, fc0, fc1))
            s = fc1
    else:
        if n_train >= nt:
            raise ValueError(
                f'N_TRAIN={n_train} must be smaller than total Nt={nt}.'
            )
        segments.append((0, n_train, n_train, nt))
    return segments


# =====================================================================
# PROBE POINT HELPERS
# =====================================================================

def default_probe_points_xy(x0, y0, nx, ny):
    """Return a default 5-point probe layout in full-domain index space.

    Generates a center point plus one point per quadrant.

    Parameters
    ----------
    x0, y0 : int — origin of the spatial subset in full-domain indices
    nx, ny  : int — width and height of the spatial subset

    Returns
    -------
    list of (x_full, y_full) int tuples
    """
    xc = x0 + nx // 2
    yc = y0 + ny // 2
    xl = x0 + nx // 4
    xr = x0 + (3 * nx) // 4
    yb = y0 + ny // 4
    yt = y0 + (3 * ny) // 4
    return [(xc, yc), (xl, yb), (xr, yb), (xl, yt), (xr, yt)]


def snap_probe_to_valid(px_full, py_full, valid_mask, x0, y0):
    """Snap a requested probe location to the nearest valid grid point.

    Parameters
    ----------
    px_full, py_full : int — probe location in full-domain index space
    valid_mask       : bool ndarray, shape (Ny, Nx) — True at valid points
    x0, y0           : int — origin of the spatial subset in full-domain indices

    Returns
    -------
    px_full, py_full : int — snapped full-domain coordinates
    pxi, pyi         : int — snapped local (subset) indices
    """
    Ny, Nx = valid_mask.shape
    pxi0 = int(px_full - x0)
    pyi0 = int(py_full - y0)

    if 0 <= pyi0 < Ny and 0 <= pxi0 < Nx and valid_mask[pyi0, pxi0]:
        return int(px_full), int(py_full), pxi0, pyi0

    best = None
    max_r = max(Nx, Ny)
    for r in range(0, max_r):
        y_min = max(0, pyi0 - r)
        y_max = min(Ny - 1, pyi0 + r)
        x_min = max(0, pxi0 - r)
        x_max = min(Nx - 1, pxi0 + r)
        for yy in range(y_min, y_max + 1):
            for xx in range(x_min, x_max + 1):
                if valid_mask[yy, xx]:
                    d2 = (yy - pyi0) ** 2 + (xx - pxi0) ** 2
                    if best is None or d2 < best[0]:
                        best = (d2, xx, yy)
        if best is not None:
            _, xx, yy = best
            return x0 + xx, y0 + yy, xx, yy

    raise RuntimeError('No valid probe points found in selected box.')


def build_probe_list(probe_points_xy, mask, x0, y0, Nx, Ny):
    """Validate and snap all requested probe points to valid grid locations.

    Parameters
    ----------
    probe_points_xy : list of (x, y) or None
        Full-domain (x, y) index pairs. If None, auto-generates 5 default points.
    mask            : bool ndarray, shape (Ny, Nx)
    x0, y0          : int — origin of the spatial subset in full-domain indices
    Nx, Ny          : int — spatial subset dimensions

    Returns
    -------
    probe_points : list of dicts with keys:
        'name'   : str
        'x_full' : int — full-domain x index
        'y_full' : int — full-domain y index
        'pxi'    : int — local x index within the spatial subset
        'pyi'    : int — local y index within the spatial subset
    """
    if probe_points_xy is None:
        probe_points_xy = default_probe_points_xy(x0, y0, Nx, Ny)
    if len(probe_points_xy) == 0:
        raise ValueError('probe_points_xy must contain at least one (x, y) point.')

    probe_points = []
    for i, (px_full, py_full) in enumerate(probe_points_xy):
        px_full, py_full, pxi, pyi = snap_probe_to_valid(
            px_full, py_full, mask, x0, y0
        )
        probe_points.append({
            'name': f'P{i + 1}',
            'x_full': int(px_full),
            'y_full': int(py_full),
            'pxi': int(pxi),
            'pyi': int(pyi),
        })
    return probe_points


# =====================================================================
# SEGMENT RUNNER
# =====================================================================

def run_method_over_segments(Xmat, Nt, segments, method_fn, *args):
    """Apply a single forecast method across all train/forecast segments.

    Assembles per-segment results into a single (n_features, Nt) array.
    Entries outside forecast windows are filled with NaN.

    Parameters
    ----------
    Xmat      : ndarray, shape (n_features, Nt) — full time-series data matrix
    Nt        : int — total number of time steps
    segments  : list of (tr0, tr1, fc0, fc1) tuples (from build_forecast_segments)
    method_fn : callable — forecast function from dmd_methods
    *args     : additional positional arguments forwarded to method_fn

    Returns
    -------
    full : ndarray, shape (n_features, Nt)
        Forecast values within forecast windows; NaN elsewhere.
    """
    full = np.full((Xmat.shape[0], Nt), np.nan)
    for tr0, tr1, fc0, fc1 in segments:
        Xtr = Xmat[:, tr0:tr1]
        nseg = fc1 - fc0
        if nseg <= 0:
            continue
        full[:, fc0:fc1] = method_fn(Xtr, nseg, *args)
    return full
