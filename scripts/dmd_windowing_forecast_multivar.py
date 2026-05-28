"""Forecast multivariable 2-D fields with several DMD variants.

WORKFLOW:
  1. Load u, v, temp, salt from NetCDF files and regrid to rho grid.
  2. Build a stacked multivariable state vector.
  3. Mask invalid/land points (intersection across all variables).
  4. Partition time series into alternating train/forecast segments.
  5. Run each DMD method over all segments via dmd_methods + dmd_utils.
  6. Extract per-variable forecast cubes and compute metrics via dmd_plotting.

METHODS (from dmd_methods.py):
  Standard DMD, Sliding-window DMD, Hankel DMD, Sliding Hankel,
  EDMD (poly), Kernel DMD, DMD (Matlab), kEDMD, SpecRKHS-Obs, ResDMD
"""

import os
import glob
import sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

# Support both normal script execution and notebook/cell execution where
# __file__ is not defined.
if '__file__' in globals():
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_candidates = [
        os.path.normpath(os.path.join(_base_dir, '..')),
        _base_dir,
    ]
else:
    _base_dir = os.getcwd()
    _repo_candidates = [
        _base_dir,
        os.path.normpath(os.path.join(_base_dir, '..')),
        os.path.normpath(os.path.join(_base_dir, 'dmd_testing')),
    ]

REPO_ROOT = next(
    (p for p in _repo_candidates if os.path.isdir(os.path.join(p, 'code'))),
    _repo_candidates[0],
)
CODE_DIR = os.path.join(REPO_ROOT, 'code')
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from dmd_methods import (
    dmd_forecast_field,
    sliding_dmd_forecast_field,
    hankel_dmd_forecast_field,
    sliding_hankel_dmd_forecast_field,
    edmd_forecast_field,
    kernel_dmd_forecast_field,
    dmd_matlab_forecast_field,
    kedmd_forecast_field,
    specrkhs_obs_forecast_field,
    resdmd_forecast_field,
)
from dmd_utils import (
    cube_to_mat,
    mat_to_cube,
    build_forecast_segments,
    build_probe_list,
    run_method_over_segments,
)
from dmd_plotting import (
    plot_probe_timeseries,
    plot_spatial_rmse_map,
    plot_snapshot_comparison,
    save_spatial_snapshot_frames,
    plot_forecast_rmse_per_step,
    print_rmse_report,
)
from cmocean import cm as cmo

# =====================================================================
# TUNABLE PARAMETERS
# =====================================================================

DATA_SOURCE = 'remora'
SCRIPT_DIR = (
    os.path.dirname(os.path.abspath(__file__))
    if '__file__' in globals()
    else os.getcwd()
)
_DATA_ROOT_CANDIDATES = [
    os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data')),
    os.path.normpath(os.path.join(SCRIPT_DIR, 'data')),
    os.path.normpath(os.path.join(os.getcwd(), '..', 'data')),
    os.path.normpath(os.path.join(os.getcwd(), 'data')),
]
DATA_ROOT = next((p for p in _DATA_ROOT_CANDIDATES if os.path.isdir(p)),
                 _DATA_ROOT_CANDIDATES[0])
DATA_DIR = os.path.join(DATA_ROOT, DATA_SOURCE)

FILE_PATTERNS_BY_VAR = {
    'u':    ['u.nc'],
    'v':    ['v.nc'],
    'temp': ['temp.nc'],
    'salt': ['salt.nc'],
}

STATE_VARIABLES = ['u', 'v', 'temp', 'salt']
DISPLAY_VAR     = 'u'     # variable used for display-only plots (e.g. probe series)

# ---- Time Subset ----
TIME_START = None
TIME_STOP  = None
TIME_STEP  = 1

# ---- Spatial Subset ----
X_SLICE = slice(None)
Y_SLICE = slice(None)

# ---- Windowing ----
N_TRAIN                 = 300
USE_ALTERNATING_WINDOWS = True
ALT_TRAIN_WINDOW        = 200
ALT_FORECAST_WINDOW     = 60
ALT_START_INDEX         = 0

# ---- Probe Points ----
PROBE_POINTS_XY = None   # None = auto-generate 5 points
SNAP_STEP       = 5

# ---- DMD Hyperparameters ----
DMD_RANK  = 20

SW_WINDOW = 120
SW_RANK   = 15
SW_STRIDE = 1

HANKEL_DELAYS = 6
HANKEL_RANK   = 25
HANKEL_SW_WIN = 200
HANKEL_SW_RNK = 20

EDMD_POD_RANK   = 15
EDMD_DELAYS     = 4
EDMD_POLY_ORDER = 2

KDMD_DELAYS = 6
KDMD_SIGMA  = None
KDMD_RANK   = 25
KDMD_REG    = 1e-8

KEDMD_SIGMA = None
KEDMD_REG   = 1e-8

SPECRKHS_SIGMA    = None
SPECRKHS_REG      = 1e-8
SPECRKHS_RES_TOL  = 0.2
SPECRKHS_MIN_KEEP = 20

RESDMD_POD_RANK   = 15
RESDMD_DELAYS     = 4
RESDMD_POLY_ORDER = 2
RESDMD_TOL        = 0.5
RESDMD_MIN_KEEP   = 8

# ---- Output ----
PLOT_OUTPUT_DIR        = '../plots/multivar'
SAVE_SPATIAL_SNAPSHOTS = True
SPATIAL_SNAPSHOT_DIR   = 'spatial_snapshots'
SPATIAL_SNAPSHOT_DPI   = 120
SPATIAL_CMAP           = cmo.balance
SPATIAL_COLOR_LIMIT    = None

# Variable-specific colormaps and fixed limits for scalar fields
VAR_SPATIAL_CMAP = {
    'temp': cmo.thermal,
    'salt': cmo.haline,
}
VAR_SPATIAL_LIMITS = {
    'temp': (0.0, 30.0),
    'salt': (30.0, 37.0),
}

# =====================================================================
# DATA LOADING HELPERS
# =====================================================================

def find_files_for_var(var_name):
    if DATA_SOURCE not in {'roms', 'remora'}:
        raise ValueError(f"DATA_SOURCE must be 'roms' or 'remora', got: {DATA_SOURCE}")
    pats = FILE_PATTERNS_BY_VAR.get(var_name, [])
    globs = [os.path.join(DATA_DIR, p) for p in pats]
    files = sorted({f for pat in globs for f in glob.glob(pat)})
    if not files:
        raise FileNotFoundError(
            f'No files matched patterns for {var_name}: {pats} in DATA_DIR={DATA_DIR}'
        )
    return files


def load_var_da(var_name):
    ds = xr.open_mfdataset(
        find_files_for_var(var_name),
        chunks={'ocean_time': 1},
        data_vars='minimal',
        coords='minimal',
        compat='override',
    )
    if var_name not in ds.data_vars:
        raise ValueError(
            f"Variable '{var_name}' not found in dataset: {list(ds.data_vars)}"
        )
    da = ds[var_name]
    if 'ocean_time' not in da.dims:
        raise ValueError(f"Variable '{var_name}' missing ocean_time dim: {da.dims}")
    isel_map = {'ocean_time': slice(TIME_START, TIME_STOP, TIME_STEP)}
    if 's_rho' in da.dims:
        isel_map['s_rho'] = -1
    elif 's_w' in da.dims:
        isel_map['s_w'] = -1
    return da.isel(**isel_map).astype(float)


def da_to_cube(da):
    y_dim = da.dims[-2]
    x_dim = da.dims[-1]
    return np.asarray(da.transpose('ocean_time', y_dim, x_dim).values, dtype=float)


def u_to_rho(u_cube, target_shape):
    """Regrid u from xi_u to rho grid when x-dimension is staggered."""
    nt, ny_u, nx_u = u_cube.shape
    _, ny_rho, nx_rho = target_shape
    if (ny_u, nx_u) == (ny_rho, nx_rho):
        return u_cube
    if ny_u == ny_rho and nx_u == nx_rho - 1:
        out = np.empty((nt, ny_u, nx_u + 1), dtype=float)
        out[:, :, 1:-1] = 0.5 * (u_cube[:, :, :-1] + u_cube[:, :, 1:])
        out[:, :, 0] = u_cube[:, :, 0]
        out[:, :, -1] = u_cube[:, :, -1]
        return out
    raise ValueError(
        f'Cannot align u to rho grid: u={u_cube.shape}, target={target_shape}'
    )


def v_to_rho(v_cube, target_shape):
    """Regrid v from eta_v to rho grid when y-dimension is staggered."""
    nt, ny_v, nx_v = v_cube.shape
    _, ny_rho, nx_rho = target_shape
    if (ny_v, nx_v) == (ny_rho, nx_rho):
        return v_cube
    if ny_v == ny_rho - 1 and nx_v == nx_rho:
        out = np.empty((nt, ny_v + 1, nx_v), dtype=float)
        out[:, 1:-1, :] = 0.5 * (v_cube[:, :-1, :] + v_cube[:, 1:, :])
        out[:, 0, :] = v_cube[:, 0, :]
        out[:, -1, :] = v_cube[:, -1, :]
        return out
    raise ValueError(
        f'Cannot align v to rho grid: v={v_cube.shape}, target={target_shape}'
    )


def get_spatial_cmap(vn):
    return VAR_SPATIAL_CMAP.get(vn, SPATIAL_CMAP)


def get_spatial_limits(vn, truth_cube_var):
    fixed = VAR_SPATIAL_LIMITS.get(vn)
    if fixed is not None:
        return float(fixed[0]), float(fixed[1])
    truth_finite = truth_cube_var[np.isfinite(truth_cube_var)]
    if SPATIAL_COLOR_LIMIT is not None:
        vmax = float(abs(SPATIAL_COLOR_LIMIT))
    elif truth_finite.size > 0:
        vmax = float(np.percentile(np.abs(truth_finite), 99))
    else:
        vmax = 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return -vmax, vmax

# =====================================================================
# LOAD DATA
# =====================================================================
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
print(f"Using data source '{DATA_SOURCE}' from {DATA_DIR}")

temp_cube_full = da_to_cube(load_var_da('temp'))
salt_cube_full = da_to_cube(load_var_da('salt'))
u_cube_full    = u_to_rho(da_to_cube(load_var_da('u')), temp_cube_full.shape)
v_cube_full    = v_to_rho(da_to_cube(load_var_da('v')), temp_cube_full.shape)

shape_ref = temp_cube_full.shape
for nm, cube in [('salt', salt_cube_full), ('u', u_cube_full), ('v', v_cube_full)]:
    if cube.shape != shape_ref:
        raise ValueError(
            f'Shape mismatch after rho conversion: temp={shape_ref}, {nm}={cube.shape}'
        )

var_cubes = {
    'u':    u_cube_full[:, Y_SLICE, X_SLICE],
    'v':    v_cube_full[:, Y_SLICE, X_SLICE],
    'temp': temp_cube_full[:, Y_SLICE, X_SLICE],
    'salt': salt_cube_full[:, Y_SLICE, X_SLICE],
}

if DISPLAY_VAR not in STATE_VARIABLES:
    raise ValueError(f'DISPLAY_VAR={DISPLAY_VAR!r} must be in STATE_VARIABLES.')

U_cube = var_cubes[DISPLAY_VAR]
Nt, Ny, Nx = U_cube.shape
print(
    f"Loaded stacked state vars {STATE_VARIABLES}; "
    f"display var '{DISPLAY_VAR}' cube (t,y,x) = {U_cube.shape}"
)

# ---- Spatial mask (intersection over all state variables) ----
mask = np.ones((Ny, Nx), dtype=bool)
for vn in STATE_VARIABLES:
    mask &= np.isfinite(var_cubes[vn]).all(axis=0)

n_valid = int(mask.sum())
print(f'Valid points across {STATE_VARIABLES}: {n_valid} / {Ny * Nx}')
if n_valid == 0:
    raise RuntimeError('No valid points for multivariable state in selected box.')

# ---- Stacked data matrix ----
X_blocks = [cube_to_mat(var_cubes[vn], mask) for vn in STATE_VARIABLES]
Xmat = np.vstack(X_blocks)   # (n_vars * n_valid, Nt)
t    = np.arange(Nt)

# ---- Helpers to extract a variable block from a stacked state matrix ----
def state_mat_to_var_cube(state_mat, vn):
    """Extract the block for variable `vn` from a stacked state matrix."""
    iv = STATE_VARIABLES.index(vn)
    i0 = iv * n_valid
    i1 = (iv + 1) * n_valid
    return mat_to_cube(state_mat[i0:i1, :], mask, Ny, Nx)

# =====================================================================
# WINDOWING
# =====================================================================
segments = build_forecast_segments(
    Nt,
    USE_ALTERNATING_WINDOWS,
    N_TRAIN,
    ALT_TRAIN_WINDOW,
    ALT_FORECAST_WINDOW,
    ALT_START_INDEX,
)
if not segments:
    raise RuntimeError('No valid forecast segments were generated.')

forecast_time_idx = np.concatenate(
    [np.arange(fc0, fc1, dtype=int) for (_, _, fc0, fc1) in segments]
)
n_fc = len(forecast_time_idx)

min_train_len = min(tr1 - tr0 for tr0, tr1, _, _ in segments)
required_train_len = max(2, HANKEL_DELAYS, EDMD_DELAYS, KDMD_DELAYS, RESDMD_DELAYS)
if min_train_len < required_train_len:
    raise ValueError(
        f'Minimum training window ({min_train_len}) < required ({required_train_len}). '
        f'Increase ALT_TRAIN_WINDOW or reduce delay settings.'
    )

# =====================================================================
# PROBE POINTS
# =====================================================================
x0 = 0 if X_SLICE.start is None else X_SLICE.start
y0 = 0 if Y_SLICE.start is None else Y_SLICE.start
probe_points = build_probe_list(PROBE_POINTS_XY, mask, x0, y0, Nx, Ny)

print('Using probe points (full-domain x, y):')
for p in probe_points:
    print(f"  {p['name']}: ({p['x_full']}, {p['y_full']})")

print('\nForecast segments (train -> forecast):')
for tr0, tr1, fc0, fc1 in segments:
    print(f'  train [{tr0}:{tr1}) -> forecast [{fc0}:{fc1})')

# =====================================================================
# RUN ALL FORECAST METHODS
# =====================================================================
print('\nRunning forecasts...')

forecasts_mat = {}

print('  DMD...')
forecasts_mat['DMD'] = run_method_over_segments(
    Xmat, Nt, segments, dmd_forecast_field, DMD_RANK)

print('  Sliding DMD...')
forecasts_mat['Sliding DMD'] = run_method_over_segments(
    Xmat, Nt, segments, sliding_dmd_forecast_field, SW_WINDOW, SW_RANK, SW_STRIDE)

print('  Hankel DMD...')
forecasts_mat['Hankel DMD'] = run_method_over_segments(
    Xmat, Nt, segments, hankel_dmd_forecast_field, HANKEL_DELAYS, HANKEL_RANK)

print('  Sliding Hankel DMD...')
forecasts_mat['Sliding Hankel'] = run_method_over_segments(
    Xmat, Nt, segments, sliding_hankel_dmd_forecast_field,
    HANKEL_SW_WIN, HANKEL_DELAYS, HANKEL_SW_RNK, SW_STRIDE)

print('  EDMD (poly)...')
forecasts_mat['EDMD (poly)'] = run_method_over_segments(
    Xmat, Nt, segments, edmd_forecast_field, EDMD_POD_RANK, EDMD_DELAYS, EDMD_POLY_ORDER)

print('  Kernel DMD...')
forecasts_mat['Kernel DMD'] = run_method_over_segments(
    Xmat, Nt, segments, kernel_dmd_forecast_field, KDMD_DELAYS, KDMD_SIGMA, KDMD_RANK, KDMD_REG)

print('  DMD (Matlab)...')
forecasts_mat['DMD (Matlab)'] = run_method_over_segments(
    Xmat, Nt, segments, dmd_matlab_forecast_field)

print('  kEDMD...')
forecasts_mat['kEDMD'] = run_method_over_segments(
    Xmat, Nt, segments, kedmd_forecast_field, KEDMD_SIGMA, KEDMD_REG)

print('  SpecRKHS-Obs...')
forecasts_mat['SpecRKHS-Obs'] = run_method_over_segments(
    Xmat, Nt, segments, specrkhs_obs_forecast_field,
    SPECRKHS_SIGMA, SPECRKHS_REG, SPECRKHS_RES_TOL, SPECRKHS_MIN_KEEP)

print('  ResDMD...')
forecasts_mat['ResDMD'] = run_method_over_segments(
    Xmat, Nt, segments, resdmd_forecast_field,
    RESDMD_POD_RANK, RESDMD_DELAYS, RESDMD_POLY_ORDER, RESDMD_TOL, RESDMD_MIN_KEEP)

# Build per-variable forecast cubes
truth_cubes_by_var = {vn: var_cubes[vn] for vn in STATE_VARIABLES}
forecasts_by_var = {
    vn: {name: state_mat_to_var_cube(f, vn) for name, f in forecasts_mat.items()}
    for vn in STATE_VARIABLES
}

# =====================================================================
# VISUALIZATION & REPORTING
# =====================================================================
print('\nSaving figures...')

# Figure 1 — probe time series per variable
for vn in STATE_VARIABLES:
    plot_probe_timeseries(
        probe_points, t, truth_cubes_by_var[vn], forecasts_by_var[vn],
        os.path.join(PLOT_OUTPUT_DIR, f'figure1_probe_timeseries_{vn}.png'),
        var_label=vn,
    )

# Figure 2 — spatial RMSE map per variable
for vn in STATE_VARIABLES:
    plot_spatial_rmse_map(
        forecasts_by_var[vn], truth_cubes_by_var[vn],
        forecast_time_idx, probe_points, n_fc,
        os.path.join(PLOT_OUTPUT_DIR, f'figure2_spatial_rmse_map_{vn}.png'),
        title_suffix=vn,
    )

# Figure 3 — snapshot comparison per variable
for vn in STATE_VARIABLES:
    vmin_s, vmax_s = get_spatial_limits(vn, truth_cubes_by_var[vn])
    plot_snapshot_comparison(
        forecasts_by_var[vn], truth_cubes_by_var[vn],
        forecast_time_idx, probe_points, SNAP_STEP,
        os.path.join(PLOT_OUTPUT_DIR, f'figure3_snapshot_comparison_{vn}.png'),
        cmap=get_spatial_cmap(vn), vmin=vmin_s, vmax=vmax_s,
        title_suffix=vn,
    )

# Figure 4 — per-step RMSE (all variables in one figure)
plot_forecast_rmse_per_step(
    [forecasts_by_var[vn] for vn in STATE_VARIABLES],
    [truth_cubes_by_var[vn] for vn in STATE_VARIABLES],
    forecast_time_idx, n_fc,
    os.path.join(PLOT_OUTPUT_DIR, 'figure4_forecast_rmse_per_step.png'),
    var_labels=STATE_VARIABLES,
)

# Spatial snapshot frames per variable
if SAVE_SPATIAL_SNAPSHOTS:
    for vn in STATE_VARIABLES:
        vmin_s, vmax_s = get_spatial_limits(vn, truth_cubes_by_var[vn])
        snap_dir = os.path.join(PLOT_OUTPUT_DIR, f'{SPATIAL_SNAPSHOT_DIR}_{vn}')
        save_spatial_snapshot_frames(
            forecasts_by_var[vn], truth_cubes_by_var[vn],
            forecast_time_idx, probe_points, snap_dir,
            cmap=get_spatial_cmap(vn), vmin=vmin_s, vmax=vmax_s,
            dpi=SPATIAL_SNAPSHOT_DPI, var_label=vn,
        )

# RMSE report per variable
for vn in STATE_VARIABLES:
    print_rmse_report(
        forecasts_by_var[vn], truth_cubes_by_var[vn],
        probe_points, forecast_time_idx, n_fc,
        var_name=vn,
    )

print('\nDone.')
