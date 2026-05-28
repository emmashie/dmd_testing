"""Forecast a 2-D velocity field u(t, y, x) with several DMD variants.

WORKFLOW:
  1. Load data from NetCDF files (configurable variable & time range).
  2. Mask invalid/land points.
  3. Partition time series into alternating train/forecast segments.
  4. Run each DMD method over all segments via dmd_methods + dmd_utils.
  5. Compute and visualize error metrics via dmd_plotting.
  6. Export spatial snapshots per forecast step.

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

DATA_SOURCE      = 'remora'
SCRIPT_DIR       = (
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
DATA_ROOT        = next((p for p in _DATA_ROOT_CANDIDATES if os.path.isdir(p)),
                        _DATA_ROOT_CANDIDATES[0])
DATA_DIR         = os.path.join(DATA_ROOT, DATA_SOURCE)
FILE_PATTERNS    = ['u.nc']
VARIABLE_CHOICES = ['u']

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
PLOT_OUTPUT_DIR        = '../plots/univar'
SAVE_SPATIAL_SNAPSHOTS = True
SPATIAL_SNAPSHOT_DIR   = 'spatial_snapshots'
SPATIAL_SNAPSHOT_DPI   = 120
SPATIAL_CMAP           = cmo.balance
SPATIAL_COLOR_LIMIT    = None   # numeric value forces +/- symmetric limit

# =====================================================================
# LOAD DATA
# =====================================================================
if DATA_SOURCE not in {'roms', 'remora'}:
    raise ValueError(f"DATA_SOURCE must be 'roms' or 'remora', got: {DATA_SOURCE}")

file_globs = [os.path.join(DATA_DIR, pat) for pat in FILE_PATTERNS]
matched_files = sorted({f for pat in file_globs for f in glob.glob(pat)})
if not matched_files:
    raise FileNotFoundError(
        f'No files matched FILE_PATTERNS in DATA_DIR.\n'
        f'DATA_DIR={DATA_DIR}, FILE_PATTERNS={FILE_PATTERNS}'
    )

ds = xr.open_mfdataset(
    matched_files,
    chunks={'ocean_time': 1},
    data_vars='minimal',
    coords='minimal',
    compat='override',
)
print(f"Using data source '{DATA_SOURCE}' from {DATA_DIR}")
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

var_name = next((v for v in VARIABLE_CHOICES if v in ds.data_vars), None)
if var_name is None:
    raise ValueError(
        f'None of VARIABLE_CHOICES={VARIABLE_CHOICES} found in dataset: '
        f'{list(ds.data_vars)}'
    )

u_da = ds[var_name]
if 'ocean_time' not in u_da.dims:
    raise ValueError(
        f"Variable '{var_name}' must have 'ocean_time' dim. Found: {u_da.dims}"
    )

isel_map = {'ocean_time': slice(TIME_START, TIME_STOP, TIME_STEP)}
if 's_rho' in u_da.dims:
    isel_map['s_rho'] = -1
y_dim = u_da.dims[-2]
x_dim = u_da.dims[-1]
isel_map[y_dim] = Y_SLICE
isel_map[x_dim] = X_SLICE

U_cube = u_da.isel(**isel_map).values.astype(float)
Nt, Ny, Nx = U_cube.shape
print(f"Loaded '{var_name}' cube (t,y,x) = {U_cube.shape}")

# ---- Spatial mask ----
mask = np.isfinite(U_cube).all(axis=0)
n_valid = int(mask.sum())
print(f'Valid (wet) points: {n_valid} / {Ny * Nx}')
if n_valid == 0:
    raise RuntimeError('No valid points in selected box.')

# ---- Data matrix ----
Xmat = cube_to_mat(U_cube, mask)   # (n_valid, Nt)
t    = np.arange(Nt)

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

# Convert (n_valid, Nt) matrices -> (Nt, Ny, Nx) cubes
forecasts_cube = {
    name: mat_to_cube(f, mask, Ny, Nx)
    for name, f in forecasts_mat.items()
}

# =====================================================================
# VISUALIZATION & REPORTING
# =====================================================================

# Colour limits for spatial snapshots
if SPATIAL_COLOR_LIMIT is not None:
    snap_vmax = float(abs(SPATIAL_COLOR_LIMIT))
else:
    truth_finite = U_cube[forecast_time_idx][np.isfinite(U_cube[forecast_time_idx])]
    snap_vmax = float(np.percentile(np.abs(truth_finite), 99)) if truth_finite.size > 0 else 1.0
    if not np.isfinite(snap_vmax) or snap_vmax <= 0:
        snap_vmax = 1.0
snap_vmin = -snap_vmax

print('\nSaving figures...')

plot_probe_timeseries(
    probe_points, t, U_cube, forecasts_cube,
    os.path.join(PLOT_OUTPUT_DIR, 'figure1_probe_timeseries.png'),
    var_label=var_name,
)

plot_spatial_rmse_map(
    forecasts_cube, U_cube, forecast_time_idx, probe_points, n_fc,
    os.path.join(PLOT_OUTPUT_DIR, 'figure2_spatial_rmse_map.png'),
)

plot_snapshot_comparison(
    forecasts_cube, U_cube, forecast_time_idx, probe_points, SNAP_STEP,
    os.path.join(PLOT_OUTPUT_DIR, 'figure3_snapshot_comparison.png'),
    cmap=SPATIAL_CMAP, vmin=snap_vmin, vmax=snap_vmax,
)

plot_forecast_rmse_per_step(
    forecasts_cube, U_cube, forecast_time_idx, n_fc,
    os.path.join(PLOT_OUTPUT_DIR, 'figure4_forecast_rmse_per_step.png'),
    var_labels=var_name,
)

if SAVE_SPATIAL_SNAPSHOTS:
    snap_out_dir = os.path.join(PLOT_OUTPUT_DIR, SPATIAL_SNAPSHOT_DIR)
    save_spatial_snapshot_frames(
        forecasts_cube, U_cube, forecast_time_idx, probe_points,
        snap_out_dir,
        cmap=SPATIAL_CMAP, vmin=snap_vmin, vmax=snap_vmax,
        dpi=SPATIAL_SNAPSHOT_DPI,
    )

print_rmse_report(
    forecasts_cube, U_cube, probe_points, forecast_time_idx, n_fc,
    var_name=var_name,
)

print('\nDone.')
