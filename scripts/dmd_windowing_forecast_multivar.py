"""Forecasting multivariable 2-D ROMS fields with several DMD variants.

DYNAMIC MODE DECOMPOSITION (DMD) COMPARISON FRAMEWORK
======================================================
This script loads spatiotemporal oceanographic data, partitions it into
alternating train/forecast windows, and applies seven different DMD variants
to predict future velocity fields. Results are compared across methods via
spatial RMSE maps, probe time series, and snapshot visualizations.

METHODS IMPLEMENTED:
  1. Standard DMD          : Classical DMD on raw data snapshots
  2. Sliding-window DMD    : DMD with window re-fitting during forecast
  3. Hankel DMD            : Time-delay embedding with fixed delays
  4. Sliding Hankel        : Hankel DMD with sliding window updates
  5. EDMD (poly)           : Extended DMD using POD + polynomial dictionary
  6. Kernel DMD            : Nonlinear DMD with Gaussian RBF kernel
  7. ResDMD                : Residual-filtered DMD (removes spurious modes)

WORKFLOW:
    - Load ROMS variables u, v, temp, salt from NetCDF files
    - Regrid u and v from C-grid staggered points to rho points
  - Mask invalid/land points
    - Build a stacked multivariable state vector
  - Partition time series into alternating train/forecast segments
  - Run each method over all segments
  - Compute and visualize error metrics
  - Export spatial snapshots per forecast step
"""
import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from numpy.linalg import pinv, svd, eig
from cmocean import cm as cmo
plt.ioff()

# =====================================================================
# TUNABLE PARAMETERS
# =====================================================================
# Configure data source, time/space selection, windowing strategy, and DMD variants.
# All parameters can be adjusted without modifying the core forecast logic.

# Select dataset source from data folder subdirectories: 'roms' or 'remora'.
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
DATA_ROOT = next((p for p in _DATA_ROOT_CANDIDATES if os.path.isdir(p)), _DATA_ROOT_CANDIDATES[0])
DATA_DIR = os.path.join(DATA_ROOT, DATA_SOURCE)

# File patterns per variable. Each variable is loaded independently and then merged.
FILE_PATTERNS_BY_VAR = {
    'u': ['u.nc'],
    'v': ['v.nc'],
    'temp': ['temp.nc'],
    'salt': ['salt.nc'],
}

# Variable order in the stacked multivariable state vector.
STATE_VARIABLES = ['u', 'v', 'temp', 'salt']

# Variable used for plotting/error maps while forecasts are driven by full state.
DISPLAY_VAR = 'u'

# ---- Time Subset Configuration ----
# Optional temporal subset of the concatenated record. Set both to None for full record.
TIME_START = None                # Start time index (None = beginning)
TIME_STOP  = None                # Stop time index (None = end)
TIME_STEP  = 1                   # Time subsampling stride

# ---- Spatial Subset Configuration ----
# Slice indices in ROMS rho-grid index space (after u/v regridding to rho).
X_SLICE = slice(None)            # Apply None to use full x-dimension
Y_SLICE = slice(None)            # Apply None to use full y-dimension

# ---- Training/Forecast Window Configuration ----
# Strategy 1: Single train/test split (when USE_ALTERNATING_WINDOWS=False)
N_TRAIN = 300                    # Length of single training block

# Strategy 2: Alternating train/forecast windows (when USE_ALTERNATING_WINDOWS=True)
# Cycle through entire time series, alternating fixed-length train/forecast blocks.
USE_ALTERNATING_WINDOWS = True   # Toggle between windowing strategies
ALT_TRAIN_WINDOW        = 200    # Training block length (time steps)
ALT_FORECAST_WINDOW     = 60     # Forecast block length (time steps)
ALT_START_INDEX         = 0      # Starting time index for first training block

# ---- Diagnostic Probe Points ----
# Optional list of probes as (x, y) full-domain indices (before slicing).
# Set to None to auto-generate 5 points: center + one per quadrant.
PROBE_POINTS_XY = None
SNAP_STEP = 5                    # Which forecast step to visualize in snapshot comparison

# ---- DMD Algorithm Hyperparameters ----
# Rank/dimension controls for each method (higher = more complex, higher cost).
DMD_RANK  = 20                   # SVD truncation rank for standard DMD

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

# Kernelized methods aligned with SpecRKHS example workflow
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

# ---- Output Configuration ----
# All figures and spatial snapshots saved to PLOT_OUTPUT_DIR.
PLOT_OUTPUT_DIR         = '../plots/multivar'  # Root directory for all plot outputs
SAVE_SPATIAL_SNAPSHOTS = True
SPATIAL_SNAPSHOT_DIR   = 'spatial_snapshots'
SPATIAL_SNAPSHOT_DPI   = 120
SPATIAL_CMAP           = cmo.balance
SPATIAL_COLOR_LIMIT    = None   # set numeric value to force +/- symmetric limit

# Variable-specific colormaps for scalar fields in snapshot-style plots.
VAR_SPATIAL_CMAP = {
    'temp': cmo.thermal,
    'salt': cmo.haline,
}

# Fixed plotting limits for scalar variables (typical ocean ranges).
# Use None to keep auto/scaled behavior.
VAR_SPATIAL_LIMITS = {
    'temp': (0.0, 30.0),
    'salt': (30.0, 37.0),
}

def get_spatial_cmap(var_name):
    return VAR_SPATIAL_CMAP.get(var_name, SPATIAL_CMAP)

def get_spatial_limits(var_name, truth_cube_var):
    fixed = VAR_SPATIAL_LIMITS.get(var_name)
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
# LOAD DATA -> multivariable rho-grid state
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
        raise ValueError(f"Variable '{var_name}' not found in dataset vars: {list(ds.data_vars)}")

    da = ds[var_name]
    if 'ocean_time' not in da.dims:
        raise ValueError(f"Variable '{var_name}' missing ocean_time dim: {da.dims}")

    isel_map = {'ocean_time': slice(TIME_START, TIME_STOP, TIME_STEP)}
    if 's_rho' in da.dims:
        isel_map['s_rho'] = -1
    elif 's_w' in da.dims:
        isel_map['s_w'] = -1

    return da.isel(**isel_map).astype(float)

def u_to_rho(u_cube, target_shape):
    """Regrid u to rho grid only when x-dimension is staggered by one cell."""
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
        f"Cannot align u to rho grid: u shape={u_cube.shape}, target rho shape={target_shape}"
    )

def v_to_rho(v_cube, target_shape):
    """Regrid v to rho grid only when y-dimension is staggered by one cell."""
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
        f"Cannot align v to rho grid: v shape={v_cube.shape}, target rho shape={target_shape}"
    )

def da_to_cube(da):
    y_dim = da.dims[-2]
    x_dim = da.dims[-1]
    return np.asarray(da.transpose('ocean_time', y_dim, x_dim).values, dtype=float)

os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
print(f"Using data source '{DATA_SOURCE}' from {DATA_DIR}")

# Load all variables and convert to rho grid
temp_cube_full = da_to_cube(load_var_da('temp'))
salt_cube_full = da_to_cube(load_var_da('salt'))
u_cube_full = u_to_rho(da_to_cube(load_var_da('u')), temp_cube_full.shape)
v_cube_full = v_to_rho(da_to_cube(load_var_da('v')), temp_cube_full.shape)

shape_ref = temp_cube_full.shape
for nm, cube in [('salt', salt_cube_full), ('u', u_cube_full), ('v', v_cube_full)]:
    if cube.shape != shape_ref:
        raise ValueError(
            f'Shape mismatch after rho conversion: temp={shape_ref}, {nm}={cube.shape}'
        )

var_cubes = {
    'u': u_cube_full[:, Y_SLICE, X_SLICE],
    'v': v_cube_full[:, Y_SLICE, X_SLICE],
    'temp': temp_cube_full[:, Y_SLICE, X_SLICE],
    'salt': salt_cube_full[:, Y_SLICE, X_SLICE],
}

if DISPLAY_VAR not in STATE_VARIABLES:
    raise ValueError(f'DISPLAY_VAR={DISPLAY_VAR} must be in STATE_VARIABLES={STATE_VARIABLES}')

U_cube = var_cubes[DISPLAY_VAR]
Nt, Ny, Nx = U_cube.shape
print(
    f"Loaded stacked state vars {STATE_VARIABLES}; display var '{DISPLAY_VAR}' "
    f"cube (t,y,x) = {U_cube.shape}"
)

# Keep only points valid for all state variables over all times
mask = np.ones((Ny, Nx), dtype=bool)
for nm in STATE_VARIABLES:
    mask &= np.isfinite(var_cubes[nm]).all(axis=0)

n_valid = int(mask.sum())
print(f'Valid points across {STATE_VARIABLES}: {n_valid} / {Ny*Nx}')
if n_valid == 0:
    raise RuntimeError('No valid points for multivariable state in selected box.')

def cube_to_mat(cube):
    """Convert (t,y,x) cube to (n_valid,t) matrix using global multivariable mask."""
    return cube.reshape(cube.shape[0], -1)[:, mask.ravel()].T

def mat_to_cube(mat):
    """Convert (n_valid,t) matrix back to (t,y,x) cube with NaNs at masked points."""
    t_dim = mat.shape[1]
    out = np.full((t_dim, Ny * Nx), np.nan)
    out[:, mask.ravel()] = mat.T
    return out.reshape(t_dim, Ny, Nx)

def state_mat_to_display_cube(state_mat):
    """Extract DISPLAY_VAR block from stacked state matrix and map to cube."""
    nfeat_per_var = n_valid
    iv = STATE_VARIABLES.index(DISPLAY_VAR)
    i0 = iv * nfeat_per_var
    i1 = (iv + 1) * nfeat_per_var
    return mat_to_cube(state_mat[i0:i1, :])

def state_mat_to_var_cube(state_mat, var_name):
    """Extract an arbitrary STATE_VARIABLES block from stacked state matrix."""
    nfeat_per_var = n_valid
    iv = STATE_VARIABLES.index(var_name)
    i0 = iv * nfeat_per_var
    i1 = (iv + 1) * nfeat_per_var
    return mat_to_cube(state_mat[i0:i1, :])

X_blocks = [cube_to_mat(var_cubes[nm]) for nm in STATE_VARIABLES]
Xmat = np.vstack(X_blocks)
t = np.arange(Nt)

def build_forecast_segments(nt, use_alternating, n_train,
                            alt_train, alt_fc, alt_start):
    """Partition time series into train/forecast blocks.
    
    Returns list of tuples (train_start, train_end, forecast_start, forecast_end).
    
    Two modes:
      - use_alternating=True: cycles through record with fixed window sizes
      - use_alternating=False: single train block at start, rest is forecast block
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
            raise ValueError(f'N_TRAIN={n_train} must be smaller than loaded Nt={nt}.')
        segments.append((0, n_train, n_train, nt))
    return segments

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
truth_cube = U_cube[forecast_time_idx]

min_train_len = min(tr1 - tr0 for tr0, tr1, _, _ in segments)
required_train_len = max(
    2,
    HANKEL_DELAYS,
    EDMD_DELAYS,
    KDMD_DELAYS,
    RESDMD_DELAYS,
)
if min_train_len < required_train_len:
    raise ValueError(
        f'Minimum training window length is {min_train_len}, '
        f'but at least {required_train_len} is required by delay settings.'
    )

y0 = 0 if Y_SLICE.start is None else Y_SLICE.start
x0 = 0 if X_SLICE.start is None else X_SLICE.start

def default_probe_points_xy(x0, y0, nx, ny):
    """Return default 5-point probe layout in full-domain index space."""
    xc = x0 + nx // 2
    yc = y0 + ny // 2
    xl = x0 + nx // 4
    xr = x0 + (3 * nx) // 4
    yb = y0 + ny // 4
    yt = y0 + (3 * ny) // 4
    return [(xc, yc), (xl, yb), (xr, yb), (xl, yt), (xr, yt)]

def snap_probe_to_valid(px_full, py_full, valid_mask, x0, y0):
    """Snap a probe to the nearest valid point in the selected slice."""
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

requested_probes = PROBE_POINTS_XY
if requested_probes is None:
    requested_probes = default_probe_points_xy(x0, y0, Nx, Ny)

if len(requested_probes) == 0:
    raise ValueError('PROBE_POINTS_XY must contain at least one (x, y) point.')

probe_points = []
for i, (px_full, py_full) in enumerate(requested_probes):
    px_full, py_full, pxi, pyi = snap_probe_to_valid(px_full, py_full, mask, x0, y0)
    probe_points.append(
        {'name': f'P{i+1}', 'x_full': int(px_full), 'y_full': int(py_full),
         'pxi': int(pxi), 'pyi': int(pyi)}
    )

print('Using probe points (full-domain x, y):')
for p in probe_points:
    print(f"  {p['name']}: ({p['x_full']}, {p['y_full']})")

# =====================================================================
# GENERIC HELPERS FOR DMD METHODS
# =====================================================================
def build_hankel_mat(X, d):
    """Construct Hankel (time-delay embedding) matrix from data.
    
    Stacks d consecutive snapshots as rows: [X[0:n], X[1:n+1], ..., X[d-1:n+d-1]]
    Input: X shape (n_features, m_samples)
    Output: shape (d*n_features, m_samples-d+1)
    Used for delay-embedding methods (Hankel DMD, EDMD, etc.)
    """
    n = X.shape[1] - d + 1
    return np.vstack([X[:, i:i + n] for i in range(d)])

def dmd_fit(X, Y, r=None):
    """Fit DMD Koopman matrix A from data pairs: X (input snapshots) and Y (output snapshots).
    
    Solves: A = Y @ X-dagger (pseudoinverse solution in POD basis).
    Returns (A_t, U_r): DMD matrix in POD space and POD basis vectors (first r modes).
    """
    U, S, Vt = svd(X, full_matrices=False)
    if r is None or r > len(S):
        r = len(S)
    U_r = U[:, :r]; S_r = S[:r]; V_r = Vt[:r, :].conj().T
    A_t = U_r.conj().T @ Y @ V_r / S_r
    return A_t, U_r

def poly_dict(x, order):
    """Build polynomial dictionary for extended DMD (EDMD).
    
    Returns feature vector: [1, x, x^2, ..., x^order] stacked as rows.
    Used by EDMD and ResDMD to capture nonlinear Koopman operators.
    """
    feats = [np.ones((1, x.shape[1]))]
    for p in range(1, order + 1):
        feats.append(x ** p)
    return np.vstack(feats)

def rbf_gram_cols(Xa, Xb, sigma=None):
    """Gaussian RBF Gram matrix between columns of Xa and Xb.

    Xa: (n_features, n_a), Xb: (n_features, n_b)
    returns K: (n_a, n_b), plus sigma used.
    """
    na = np.sum(Xa * Xa, axis=0)[:, None]
    nb = np.sum(Xb * Xb, axis=0)[None, :]
    d2 = np.maximum(na + nb - 2.0 * (Xa.T @ Xb), 0.0)
    if sigma is None:
        nz = d2[d2 > 0]
        sigma = np.sqrt(np.median(nz) + 1e-12) if nz.size else 1.0
    K = np.exp(-d2 / (2.0 * sigma * sigma))
    return K, sigma

def generalized_eig_stable(A, G, reg=1e-8):
    """Solve A v = lambda G v using regularized linear solve."""
    n = G.shape[0]
    G_reg = G + reg * np.eye(n)
    M = np.linalg.solve(G_reg, A)
    vals, vecs = eig(M)
    return vals, vecs

# =====================================================================
# FORECAST METHODS
# =====================================================================
# All forecast methods accept training data (n_valid, n_train) and return
# predictions over n_fc forecast steps: shape (n_valid, n_fc)
# Forecast values are set to NaN outside requested forecast windows.

def dmd_forecast_field(Xtr, n_fc, r=None):
    """Standard DMD: fit once on training data, iterate forward using Koopman operator.
    
    Simplest method; assumes linear dynamics. Sensitive to rank choice.
    """
    X, Y = Xtr[:, :-1], Xtr[:, 1:]
    A_t, U_r = dmd_fit(X, Y, r)
    z = U_r.conj().T @ Xtr[:, -1]
    out = np.zeros((Xtr.shape[0], n_fc))
    for i in range(n_fc):
        z = A_t @ z
        out[:, i] = np.real(U_r @ z)
    return out

def sliding_dmd_forecast_field(Xtr, n_fc, window, r=None, stride=1):
    """Sliding-window DMD: re-fit every `stride` steps with most recent `window` snapshots.
    
    Adapts to time-varying dynamics by periodically re-training on recent data.
    Key parameters:
      - window: number of recent snapshots to use for each DMD fit
      - stride: how often to re-fit (stride=1 means every step, higher = more efficient)
    Often captures evolving modes better than standard DMD but costs more computationally.
    """
    hist = Xtr.copy()
    nf = Xtr.shape[0]
    out = np.zeros((nf, n_fc))
    A_t = U_r = None
    for i in range(n_fc):
        if i % stride == 0 or A_t is None:
            w = hist[:, -window:]
            A_t, U_r = dmd_fit(w[:, :-1], w[:, 1:], r)
        z = U_r.conj().T @ hist[:, -1]
        z = A_t @ z
        nxt = np.real(U_r @ z)
        out[:, i] = nxt
        hist = np.concatenate([hist, nxt[:, None]], axis=1)
    return out

def hankel_dmd_forecast_field(Xtr, n_fc, d, r=None):
    """Hankel DMD: uses time-delay embedding to expose hidden linear structure.
    
    Delay parameter d captures temporal correlations; often more efficient than standard DMD.
    """
    H = build_hankel_mat(Xtr, d)
    A_t, U_r = dmd_fit(H[:, :-1], H[:, 1:], r)
    z = U_r.conj().T @ H[:, -1]
    nf = Xtr.shape[0]
    out = np.zeros((nf, n_fc))
    for i in range(n_fc):
        z = A_t @ z
        x_full = np.real(U_r @ z)
        out[:, i] = x_full[-nf:]
    return out

def sliding_hankel_dmd_forecast_field(Xtr, n_fc, window, d, r=None, stride=1):
    """Sliding Hankel DMD: combines sliding window adaptation with delay embedding.
    
    Periodically re-fits Hankel DMD on most recent `window` snapshots.
    Balances computational cost and adaptability better than standard methods.
    
    Key tuning:
      - window: length of recent data for each DMD fit
      - d: delay dimension in Hankel embedding
      - stride: how often to re-fit
    """
    hist = Xtr.copy()
    nf = Xtr.shape[0]
    out = np.zeros((nf, n_fc))
    A_t = U_r = None
    for i in range(n_fc):
        if i % stride == 0 or A_t is None:
            w = hist[:, -window:]
            H = build_hankel_mat(w, d)
            A_t, U_r = dmd_fit(H[:, :-1], H[:, 1:], r)
        H_last = build_hankel_mat(hist[:, -d:], d)
        z = U_r.conj().T @ H_last[:, -1]
        z = A_t @ z
        x_full = np.real(U_r @ z)
        nxt = x_full[-nf:]
        out[:, i] = nxt
        hist = np.concatenate([hist, nxt[:, None]], axis=1)
    return out

def edmd_forecast_field(Xtr, n_fc, pod_rank, d, order):
    """Extended DMD (EDMD): POD reduction + Hankel delay embedding + polynomial dictionary.
    
    Three-step approach to capture weakly nonlinear Koopman operator:
      1. Apply POD to reduce dimension to pod_rank basis vectors
      2. Delay-embed POD coefficients with d delays (Hankel matrix)
      3. Fit polynomial Koopman operator using polynomial dictionary of order `order`
    
    Often provides best balance of accuracy and computational efficiency.
    Polynomial coefficients allow capturing nonlinear mode interactions.
    
    Key tuning:
      - pod_rank: initial dimension reduction (lower = cheaper, may lose structure)
      - d: time-delay embedding dimension (higher = more temporal context)
      - order: polynomial dictionary order (higher = more nonlinearity captured)
    """
    U, S, Vt = svd(Xtr, full_matrices=False)
    r = min(pod_rank, len(S))
    U_r = U[:, :r]
    A_coef = U_r.T @ Xtr
    scale = np.max(np.abs(A_coef)) + 1e-12
    A_s = A_coef / scale
    H = build_hankel_mat(A_s, d)
    Phi_X = poly_dict(H[:, :-1], order)
    Phi_Y = poly_dict(H[:, 1:],  order)
    K = Phi_Y @ pinv(Phi_X)
    B = H[:, :-1] @ pinv(Phi_X)
    phi = poly_dict(H[:, -1:], order)
    nf = Xtr.shape[0]
    out = np.zeros((nf, n_fc))
    for i in range(n_fc):
        phi = K @ phi
        z_full = np.real(B @ phi)[:, 0]
        coef_now = z_full[-r:] * scale
        out[:, i] = U_r @ coef_now
    return out

def kernel_dmd_forecast_field(Xtr, n_fc, d, sigma=None, r=15, reg=1e-8):
    """Kernel DMD: Gaussian RBF kernel applied to delay-embedded snapshots.
    
    Captures strongly nonlinear dynamics by lifting data into kernel Hilbert space.
    Uses Hankel delay embedding (d snapshots stacked) to expose hidden structure,
    then applies RBF kernel to measure similarity in delay-embedded space.
    
    Key tuning:
      - d: delay dimension (higher = more temporal context, more cost)
      - sigma: RBF bandwidth (auto-computed from median pairwise distance if None)
      - r: rank in kernel feature space
      - reg: Tikhonov regularization (prevents overfitting in decoding)
    
    Often best for capturing transient or chaotic dynamics, but expensive.
    """
    H = build_hankel_mat(Xtr, d)           # (d*nf, nt-d+1)
    nf = Xtr.shape[0]
    X = H[:, :-1].T                        # (m, d*nf)
    Y = H[:, 1:].T

    def sqdist(A, B):
        return (np.sum(A**2, 1)[:, None]
                + np.sum(B**2, 1)[None, :]
                - 2 * A @ B.T)

    D2 = sqdist(X, X)
    if sigma is None:
        sigma = np.sqrt(np.median(D2[D2 > 0]) + 1e-12)
    Gxx = np.exp(-D2 / (2 * sigma**2))
    Gyx = np.exp(-sqdist(Y, X) / (2 * sigma**2))

    evals, Q = np.linalg.eigh(Gxx)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]; Q = Q[:, idx]
    rr = int(min(r, np.sum(evals > 1e-10)))
    evals = evals[:rr]; Q = Q[:, :rr]
    Sig  = np.sqrt(np.clip(evals, 1e-12, None))
    Sinv = np.diag(1.0 / Sig)

    K_hat = Sinv @ Q.T @ Gyx @ Q @ Sinv

    # initial feature coordinates for the last training snapshot
    g0 = np.exp(-sqdist(X[-1:], X) / (2 * sigma**2))[0]
    z  = Sinv @ Q.T @ g0

    # decoder from feature space to delay-embedded state
    Z = Sinv @ Q.T @ Gxx                   # (rr, m)
    C = X.T @ Z.T @ np.linalg.inv(Z @ Z.T + reg * np.eye(rr))   # (d*nf, rr)

    out = np.zeros((nf, n_fc))
    for i in range(n_fc):
        z = K_hat @ z
        x_full = np.real(C @ z)            # (d*nf,)
        out[:, i] = x_full[-nf:]
    return out

def dmd_matlab_forecast_field(Xtr, n_fc):
    """DMD forecast matching the MATLAB antarctic/sealevel workflow.

    Uses reduced coordinates from SVD, eigendecomposition of the reduced map,
    and modal reconstruction in the same style as the linked MATLAB scripts.
    """
    x = Xtr[:, :-1]
    y = Xtr[:, 1:]
    nf = Xtr.shape[0]

    U, S, _ = svd(x, full_matrices=False)
    r = int(np.sum(S > 1e-12))
    r = max(r, 1)
    U = U[:, :r]

    PXs = x.T @ U
    PYs = y.T @ U
    K = np.linalg.lstsq(PXs, PYs, rcond=None)[0]

    lam, W = eig(K)
    PXr = PXs @ W
    PYr = PYs @ W

    rhs = np.hstack([x, y[:, -1:]]).T
    c = np.linalg.lstsq(np.vstack([PXr[0:1, :], PYr]), rhs, rcond=None)[0]

    out = np.zeros((nf, n_fc))
    for i in range(n_fc):
        weights = lam ** (i + 1)
        out[:, i] = np.real((PYr[-1, :] * weights) @ c)
    return out

def kedmd_forecast_field(Xtr, n_fc, sigma=None, reg=1e-8):
    """kEDMD forecast matching the MATLAB eig(A,G) Koopman mode workflow."""
    x = Xtr[:, :-1]
    y = Xtr[:, 1:]
    x0 = Xtr[:, -1]
    nf, m = x.shape

    G, sigma_used = rbf_gram_cols(x, x, sigma)
    A, _ = rbf_gram_cols(y, x, sigma_used)

    lam, W = generalized_eig_stable(A, G, reg=reg)

    G_start, _ = rbf_gram_cols(x0[:, None], x, sigma_used)
    B = np.vstack([G, G_start]) @ W
    targets = np.hstack([x, y[:, -1:]]).T
    mode_full = np.linalg.lstsq(B, targets, rcond=None)[0].T
    psi0_full = (G_start @ W).ravel()

    out = np.zeros((nf, n_fc))
    for i in range(n_fc):
        wts = psi0_full * (np.conj(lam) ** (i + 1))
        out[:, i] = np.real(wts @ mode_full.T)
    return out

def specrkhs_obs_forecast_field(Xtr, n_fc, sigma=None, reg=1e-8,
                                res_tol=0.2, min_keep=20):
    """SpecRKHS-Obs style forecast using kernel generalized eigenpairs.

    Mirrors the observable-evolution structure in the MATLAB examples:
      - build kernel matrices G, A, R from snapshot pairs,
      - compute generalized Koopman eigenpairs,
      - filter by residuals (SpecRKHS-inspired),
      - evolve the observable expansion from the last training snapshot.
    """
    x = Xtr[:, :-1]
    y = Xtr[:, 1:]
    x0 = Xtr[:, -1]
    nf = Xtr.shape[0]

    G, sigma_used = rbf_gram_cols(x, x, sigma)
    A, _ = rbf_gram_cols(y, x, sigma_used)
    R, _ = rbf_gram_cols(y, y, sigma_used)

    lam, F = generalized_eig_stable(A, G, reg=reg)

    # Residual proxy used for filtering: ||A v - lambda G v|| / ||R v||
    res = np.zeros(len(lam))
    for j, lj in enumerate(lam):
        v = F[:, j]
        num = np.linalg.norm((A - lj * G) @ v)
        den = np.linalg.norm(R @ v) + 1e-30
        res[j] = num / den

    keep = np.where(res < res_tol)[0]
    if keep.size < min_keep:
        keep = np.argsort(res)[:min(min_keep, len(lam))]
    lam_k = lam[keep]
    F_k = F[:, keep]

    Kx0_vals, _ = rbf_gram_cols(x0[:, None], x, sigma_used)
    Kx0_vals = Kx0_vals.ravel()

    coefs = np.linalg.lstsq((G @ F_k), Kx0_vals, rcond=None)[0]
    modes = F_k.conj().T @ x.T

    out = np.zeros((nf, n_fc))
    for i in range(n_fc):
        wts = np.conj(coefs) * (np.conj(lam_k) ** (i + 1))
        out[:, i] = np.real(wts @ modes)

    print(f'[SpecRKHS-Obs] kept {len(keep)} / {len(lam)} eigenpairs '
          f'(min res = {res.min():.2e}, max kept = {res[keep].max():.2e})')
    return out

def resdmd_forecast_field(Xtr, n_fc, pod_rank, d, order, tol, min_keep=1):
    """Residual DMD: filters spurious Koopman eigenpairs by residual error threshold.

    Combines EDMD framework with residual-based quality filtering:
      1. Reduce to POD basis
      2. Delay-embed and apply polynomial Koopman operator (like EDMD)
      3. Compute residual error for each Koopman eigenpair
      4. Keep only low-residual pairs (reliable modes)
      5. Forecast using filtered spectral expansion
    
    Removes spurious modes that fit noise rather than true dynamics.
    Often produces sharp forecasts (captures key modes only).
    """
    # 1. POD reduction
    U, S, Vt = svd(Xtr, full_matrices=False)
    r = min(pod_rank, len(S))
    U_r = U[:, :r]
    A_coef = U_r.T @ Xtr
    scale  = np.max(np.abs(A_coef)) + 1e-12
    A_s    = A_coef / scale

    # 2. Delay embedding + polynomial dictionary
    H     = build_hankel_mat(A_s, d)
    Phi_X = poly_dict(H[:, :-1], order)
    Phi_Y = poly_dict(H[:, 1:],  order)
    m     = Phi_X.shape[1]

    G = (Phi_X @ Phi_X.conj().T) / m
    A = (Phi_X @ Phi_Y.conj().T) / m
    L = (Phi_Y @ Phi_Y.conj().T) / m

    # 3. Koopman matrix and eigendecomposition
    K = Phi_Y @ pinv(Phi_X)
    evals, V = eig(K)

    # 4. Residuals for each eigenpair
    res = np.zeros(len(evals))
    for j, lam in enumerate(evals):
        v   = V[:, j]
        M1  = L - np.conj(lam) * A.conj().T - lam * A + (abs(lam) ** 2) * G
        num = np.real(v.conj() @ M1 @ v)
        den = np.real(v.conj() @ G @ v)
        res[j] = np.sqrt(max(num, 0.0) / max(den, 1e-30))

    keep = np.where(res < tol)[0]
    if keep.size < min_keep:
        keep = np.argsort(res)[:min_keep]
    lam_k = evals[keep]
    V_k   = V[:, keep]

    # 5. Decoder: features -> delay-embedded coefficients
    B = H[:, :-1] @ pinv(Phi_X)

    # 6. Initial spectral coordinates
    phi0 = poly_dict(H[:, -1:], order)[:, 0]
    c0, *_ = np.linalg.lstsq(V_k, phi0, rcond=None)

    nf = Xtr.shape[0]
    out = np.zeros((nf, n_fc))
    c = c0.astype(complex)
    for i in range(n_fc):
        c = lam_k * c
        phi    = V_k @ c
        z_full = np.real(B @ phi)
        coef_now = z_full[-r:] * scale
        out[:, i] = U_r @ coef_now
    print(f'[ResDMD] kept {len(keep)} / {len(evals)} eigenpairs '
          f'(min res = {res.min():.2e}, max kept = {res[keep].max():.2e})')
    return out

# =====================================================================
# RUN ALL FORECASTS OVER SEGMENTS
# =====================================================================
# For each train/forecast segment pair, run all 7 DMD methods.
# Collect results in (n_valid, Nt) matrices with NaN outside forecast windows.
# Then convert back to (Nt, Ny, Nx) cubes for visualization and error analysis.
forecasts_mat  = {}        # (n_valid, Nt), NaN outside forecast windows
forecasts_cube = {}        # (Nt, Ny, Nx), NaN outside forecast windows

def run_method_over_segments(method_fn, *args):
    """Apply a single forecast method across all train/forecast segments.
    
    Assembles results into a single (n_valid, Nt) array with results inside
    forecast windows and NaN elsewhere.
    """
    full = np.full((Xmat.shape[0], Nt), np.nan)
    for tr0, tr1, fc0, fc1 in segments:
        Xtr = Xmat[:, tr0:tr1]
        nseg = fc1 - fc0
        if nseg <= 0:
            continue
        full[:, fc0:fc1] = method_fn(Xtr, nseg, *args)
    return full

print('\nRunning forecasts...')
print('Forecast segments (train -> forecast):')
for tr0, tr1, fc0, fc1 in segments:
    print(f'  train [{tr0}:{tr1}) -> forecast [{fc0}:{fc1})')

print('  DMD...')
forecasts_mat['DMD'] = run_method_over_segments(
    dmd_forecast_field, DMD_RANK)

print('  Sliding DMD...')
forecasts_mat['Sliding DMD'] = run_method_over_segments(
    sliding_dmd_forecast_field, SW_WINDOW, SW_RANK, SW_STRIDE)

print('  Hankel DMD...')
forecasts_mat['Hankel DMD'] = run_method_over_segments(
    hankel_dmd_forecast_field, HANKEL_DELAYS, HANKEL_RANK)

print('  Sliding Hankel DMD...')
forecasts_mat['Sliding Hankel'] = run_method_over_segments(
    sliding_hankel_dmd_forecast_field,
    HANKEL_SW_WIN, HANKEL_DELAYS, HANKEL_SW_RNK, SW_STRIDE)

print('  EDMD (poly)...')
forecasts_mat['EDMD (poly)'] = run_method_over_segments(
    edmd_forecast_field, EDMD_POD_RANK, EDMD_DELAYS, EDMD_POLY_ORDER)

print('  Kernel DMD...')
forecasts_mat['Kernel DMD'] = run_method_over_segments(
    kernel_dmd_forecast_field, KDMD_DELAYS, KDMD_SIGMA, KDMD_RANK, KDMD_REG)

print('  DMD (Matlab)...')
forecasts_mat['DMD (Matlab)'] = run_method_over_segments(
    dmd_matlab_forecast_field)

print('  kEDMD...')
forecasts_mat['kEDMD'] = run_method_over_segments(
    kedmd_forecast_field, KEDMD_SIGMA, KEDMD_REG)

print('  SpecRKHS-Obs...')
forecasts_mat['SpecRKHS-Obs'] = run_method_over_segments(
    specrkhs_obs_forecast_field,
    SPECRKHS_SIGMA, SPECRKHS_REG,
    SPECRKHS_RES_TOL, SPECRKHS_MIN_KEEP)

print('  ResDMD...')
forecasts_mat['ResDMD'] = run_method_over_segments(
    resdmd_forecast_field,
    RESDMD_POD_RANK, RESDMD_DELAYS,
    RESDMD_POLY_ORDER, RESDMD_TOL, RESDMD_MIN_KEEP)

# Convert each method's (n_valid, Nt) forecast -> (Nt, Ny, Nx) cube
for name, f in forecasts_mat.items():
    forecasts_cube[name] = state_mat_to_display_cube(f)

# Build per-variable forecast cubes and truth cubes for all STATE_VARIABLES
# forecasts_by_var[var][method] = (Nt, Ny, Nx) cube
truth_cubes_by_var = {}
forecasts_by_var   = {}
for vn in STATE_VARIABLES:
    truth_cubes_by_var[vn] = var_cubes[vn]   # already time/space sliced
    forecasts_by_var[vn]   = {name: state_mat_to_var_cube(f, vn)
                              for name, f in forecasts_mat.items()}

# =====================================================================
# VISUALIZATION & ANALYSIS
# =====================================================================
# Generate 4 figures comparing forecast methods:
# 1. Multi-probe time series (separate panel per probe)
# 2. Spatial RMSE map per method (grid-by-grid error)
# 3. Snapshot comparison at one forecast step (spatial fields)
# 4. Per-step forecast RMSE (error evolution in time)
# Plus spatial snapshot frames (one PNG per forecast step)

# =====================================================================
# FIGURE 1 — Multi-probe time series (one figure per variable)
# =====================================================================
# One panel per probe point, repeated for each state variable.
n_probe = len(probe_points)
for vn in STATE_VARIABLES:
    truth_cube_vn = truth_cubes_by_var[vn]
    fc_by_method_vn = forecasts_by_var[vn]

    fig1_h = max(3.0 * n_probe, 6.0)
    fig1, axes1 = plt.subplots(n_probe, 1, figsize=(12, fig1_h), sharex=True, squeeze=False)
    axes1 = axes1.ravel()

    for ip, p in enumerate(probe_points):
        ax = axes1[ip]
        pxi = p['pxi']
        pyi = p['pyi']
        ax.plot(t, truth_cube_vn[:, pyi, pxi], 'k', lw=2, label='observed')
        for name, fc in fc_by_method_vn.items():
            ax.plot(t, fc[:, pyi, pxi], lw=1.2, label=name)
        ax.set_ylabel(vn)
        ax.set_title(f"{p['name']} (x={p['x_full']}, y={p['y_full']})")
        if ip == 0:
            ax.legend(ncol=4, fontsize=8, loc='best')

    axes1[-1].set_xlabel('time index')
    fig1.suptitle(f'Probe time series — {vn}', fontsize=12)
    plt.tight_layout()
    fig1.savefig(
        os.path.join(PLOT_OUTPUT_DIR, f'figure1_probe_timeseries_{vn}.png'), dpi=150
    )
    plt.close(fig1)

# =====================================================================
# FIGURE 2 — Spatial RMSE map per method (one figure per variable)
# =====================================================================
method_names = list(forecasts_cube.keys())
nm = len(method_names)
ncols = 4
nrows = int(np.ceil(nm / ncols))

for vn in STATE_VARIABLES:
    truth_cube_vn = truth_cubes_by_var[vn]
    fc_by_method_vn = forecasts_by_var[vn]

    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                               squeeze=False)

    rmse_maps = {}
    for name, fc in fc_by_method_vn.items():
        err = fc[forecast_time_idx] - truth_cube_vn[forecast_time_idx]
        rmse_maps[name] = np.sqrt(np.nanmean(err ** 2, axis=0))

    vmax2 = max(np.nanmax(r) for r in rmse_maps.values())
    vmin2 = 0.0

    for k, name in enumerate(method_names):
        ax = axes2[k // ncols, k % ncols]
        im = ax.imshow(rmse_maps[name], origin='lower',
                       vmin=vmin2, vmax=vmax2, cmap='viridis')
        ax.set_title(f'{name}  (mean={np.nanmean(rmse_maps[name]):.3e})',
                     fontsize=9)
        for p in probe_points:
            ax.plot(p['pxi'], p['pyi'], 'wx', ms=6, mew=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for k in range(nm, nrows * ncols):
        axes2[k // ncols, k % ncols].axis('off')

    fig2.suptitle(f'Spatial RMSE — {vn} (averaged over {n_fc} forecast steps)',
                  fontsize=12)
    plt.tight_layout()
    fig2.savefig(
        os.path.join(PLOT_OUTPUT_DIR, f'figure2_spatial_rmse_map_{vn}.png'), dpi=150
    )
    plt.close(fig2)

# =====================================================================
# FIGURE 3 — Snapshot comparison at step SNAP_STEP (one figure per variable)
# =====================================================================
snap = int(np.clip(SNAP_STEP, 0, n_fc - 1))
snap_t = int(forecast_time_idx[snap])

npanels = nm + 1
ncols3  = 4
nrows3  = int(np.ceil(npanels / ncols3))

for vn in STATE_VARIABLES:
    truth_cube_vn = truth_cubes_by_var[vn]
    fc_by_method_vn = forecasts_by_var[vn]
    truth_snap = truth_cube_vn[snap_t]
    cmap_vn = get_spatial_cmap(vn)
    tmin, tmax = get_spatial_limits(vn, truth_cube_vn)

    fig3, axes3 = plt.subplots(nrows3, ncols3,
                               figsize=(4 * ncols3, 3.5 * nrows3),
                               squeeze=False)

    ax = axes3[0, 0]
    im = ax.imshow(truth_snap, origin='lower', vmin=tmin, vmax=tmax, cmap=cmap_vn)
    ax.set_title(f'Truth (forecast step {snap}, t={snap_t})', fontsize=9)
    for p in probe_points:
        ax.plot(p['pxi'], p['pyi'], 'kx', ms=6, mew=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for k, name in enumerate(method_names, start=1):
        ax = axes3[k // ncols3, k % ncols3]
        im = ax.imshow(fc_by_method_vn[name][snap_t], origin='lower',
                   vmin=tmin, vmax=tmax, cmap=cmap_vn)
        ax.set_title(name, fontsize=9)
        for p in probe_points:
            ax.plot(p['pxi'], p['pyi'], 'kx', ms=6, mew=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for k in range(npanels, nrows3 * ncols3):
        axes3[k // ncols3, k % ncols3].axis('off')

    fig3.suptitle(f'{vn} — Forecast snapshot at step {snap} (t index {snap_t})',
                  fontsize=12)
    plt.tight_layout()
    fig3.savefig(
        os.path.join(PLOT_OUTPUT_DIR, f'figure3_snapshot_comparison_{vn}.png'), dpi=150
    )
    plt.close(fig3)

# =====================================================================
# SAVE 2D SPATIAL SNAPSHOT FRAMES  (one PNG per forecast step, per variable)
# =====================================================================
# Export spatial field snapshots for all state variables.
# Each variable gets its own subdirectory.
# Set SAVE_SPATIAL_SNAPSHOTS=False to skip.
if SAVE_SPATIAL_SNAPSHOTS:
    method_names_snap = list(forecasts_cube.keys())
    nm_snap = len(method_names_snap)
    ncols_s = min(4, nm_snap + 1)
    nrows_s = int(np.ceil((nm_snap + 1) / ncols_s))

    for vn in STATE_VARIABLES:
        truth_cube_vn = truth_cubes_by_var[vn]
        fc_by_method_vn = forecasts_by_var[vn]
        cmap_vn = get_spatial_cmap(vn)
        vmin_s, vmax_s = get_spatial_limits(vn, truth_cube_vn)

        out_dir = os.path.join(PLOT_OUTPUT_DIR, SPATIAL_SNAPSHOT_DIR + f'_{vn}')
        os.makedirs(out_dir, exist_ok=True)
        print(f'Saving {n_fc} spatial snapshot frames for {vn} to: {out_dir}')

        for i, tt in enumerate(forecast_time_idx):
            fig_s, axes_s = plt.subplots(
                nrows_s, ncols_s,
                figsize=(4 * ncols_s, 3.5 * nrows_s),
                squeeze=False,
            )

            ax = axes_s[0, 0]
            im = ax.imshow(truth_cube_vn[tt], origin='lower',
                           vmin=vmin_s, vmax=vmax_s, cmap=cmap_vn)
            ax.set_title(f'Truth  (t={tt})', fontsize=9)
            for p in probe_points:
                ax.plot(p['pxi'], p['pyi'], 'kx', ms=5, mew=1.2)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            for k, name in enumerate(method_names_snap, start=1):
                ax = axes_s[k // ncols_s, k % ncols_s]
                im = ax.imshow(fc_by_method_vn[name][tt], origin='lower',
                               vmin=vmin_s, vmax=vmax_s, cmap=cmap_vn)
                ax.set_title(name, fontsize=9)
                for p in probe_points:
                    ax.plot(p['pxi'], p['pyi'], 'kx', ms=5, mew=1.2)
                ax.set_xticks([]); ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            for k in range(nm_snap + 1, nrows_s * ncols_s):
                axes_s[k // ncols_s, k % ncols_s].axis('off')

            fig_s.suptitle(f'{vn} — Forecast step {i:05d}  (t index {tt})', fontsize=11)
            fig_s.tight_layout()
            fpath = os.path.join(out_dir, f'{i:05d}.png')
            fig_s.savefig(fpath, dpi=SPATIAL_SNAPSHOT_DPI)
            plt.close(fig_s)
            print(f'  [{vn}] saved frame {i + 1}/{n_fc}', end='\r')

        print(f'\nDone — {n_fc} frames for {vn} saved to: {out_dir}')

# =====================================================================
# FIGURE 4 — Per-step spatial RMSE time series (one panel per variable)
# =====================================================================
# Grid of subplots — one per state variable — showing per-step spatial mean RMSE.
nvars = len(STATE_VARIABLES)
fig4, axes4 = plt.subplots(nvars, 1, figsize=(11, 4 * nvars), sharex=True, squeeze=False)

for iv, vn in enumerate(STATE_VARIABLES):
    ax4 = axes4[iv, 0]
    truth_cube_vn = truth_cubes_by_var[vn]
    for name, fc in forecasts_by_var[vn].items():
        err = fc[forecast_time_idx] - truth_cube_vn[forecast_time_idx]
        rmse_per_step = np.sqrt(np.nanmean(err ** 2, axis=(1, 2)))
        ax4.plot(range(n_fc), rmse_per_step, '-o', lw=1.5, ms=4, label=name)
    ax4.set_ylabel(f'RMSE ({vn})')
    ax4.set_title(f'{vn} — per-step spatial mean RMSE')
    ax4.legend(loc='best', fontsize=8, ncol=3)
    ax4.grid(True, alpha=0.3)

axes4[-1, 0].set_xlabel('Forecast step')
plt.tight_layout()
fig4.savefig(os.path.join(PLOT_OUTPUT_DIR, 'figure4_forecast_rmse_per_step.png'), dpi=150)
plt.close(fig4)

# =====================================================================
# RMSE REPORT
# =====================================================================
# =====================================================================
# RMSE REPORT
# =====================================================================
# Print summary statistics comparing forecast skill across all methods.
# - Field RMSE: error averaged over space (all wet points) and time (all forecast steps)
# - Probe RMSE: error averaged across configured probe-point set
# Lower RMSE is better. Per-step table shows how error grows with forecast lead time.
print('\nForecast RMSE (space+time averaged over wet points):')
for vn in STATE_VARIABLES:
    truth_cube_vn = truth_cubes_by_var[vn]
    print(f'  Variable: {vn}')
    for name, fc in forecasts_by_var[vn].items():
        err = fc - truth_cube_vn
        rmse_full = float(np.sqrt(np.nanmean(err ** 2)))
        probe_sqerr = []
        for p in probe_points:
            pyi = p['pyi']
            pxi = p['pxi']
            probe_sqerr.append((fc[:, pyi, pxi] - truth_cube_vn[:, pyi, pxi]) ** 2)
        rmse_probe = float(np.sqrt(np.nanmean(np.stack(probe_sqerr, axis=0))))
        print(f'    {name:16s}  field RMSE = {rmse_full:.4e}   '
              f'probe-set RMSE = {rmse_probe:.4e}')

# Per-step (time-resolved) field RMSE for each variable
for vn in STATE_VARIABLES:
    truth_cube_vn = truth_cubes_by_var[vn]
    print(f'\nPer-step field RMSE — {vn} (first 5 forecast steps):')
    header = '  step  ' + '  '.join(f'{n:>14s}' for n in forecasts_by_var[vn].keys())
    print(header)
    for i in range(min(5, n_fc)):
        tt = int(forecast_time_idx[i])
        row = f'  {i:4d} (t={tt:4d})  '
        for name, fc in forecasts_by_var[vn].items():
            e = fc[tt] - truth_cube_vn[tt]
            row += f'  {np.sqrt(np.nanmean(e**2)):14.4e}'
        print(row)

print('\nDone.')