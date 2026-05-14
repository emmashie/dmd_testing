"""Forecasting a 2-D velocity field u(t, y, x) with several DMD variants.

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
  - Load data from NetCDF files (configurable variable & time range)
  - Mask invalid/land points
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

DATA_DIR         = '../data'
FILE_PATTERNS    = ['u*.nc']     # Glob patterns to match files in DATA_DIR (e.g., ['u*.nc', 'v*.nc'])
VARIABLE_CHOICES = ['u']         # Variable names to search for (in priority order; first match used)

# ---- Time Subset Configuration ----
# Optional temporal subset of the concatenated record. Set both to None for full record.
TIME_START = None                # Start time index (None = beginning)
TIME_STOP  = None                # Stop time index (None = end)
TIME_STEP  = 1                   # Time subsampling stride

# ---- Spatial Subset Configuration ----
# Slice indices in ROMS horizontal grid (eta_u/xi_u for u-component).
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

# ---- Diagnostic Point (Probe) Location ----
# Grid indices for time-series extraction (in full-domain before slicing).
PROBE_X   = 250                  # X coordinate (xi dimension)
PROBE_Y   = 100                  # Y coordinate (eta dimension)
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

RESDMD_POD_RANK   = 15
RESDMD_DELAYS     = 4
RESDMD_POLY_ORDER = 2
RESDMD_TOL        = 0.5
RESDMD_MIN_KEEP   = 8

# ---- Output Configuration ----
# All figures and spatial snapshots saved to PLOT_OUTPUT_DIR.
PLOT_OUTPUT_DIR         = '../plots'  # Root directory for all plot outputs
SAVE_SPATIAL_SNAPSHOTS = True
SPATIAL_SNAPSHOT_DIR   = 'spatial_snapshots'
SPATIAL_SNAPSHOT_DPI   = 120
SPATIAL_CMAP           = cmo.balance
SPATIAL_COLOR_LIMIT    = None   # set numeric value to force +/- symmetric limit

# =====================================================================
# LOAD DATA -> (t, y, x)
# =====================================================================
# 1. Expand file patterns using glob and concatenate all matched files.
# 2. Select requested variable (first match from VARIABLE_CHOICES).
# 3. Apply optional time/space subsetting (masking invalid/land points).
# 4. Extract to numpy array and convert to 2D matrix format for DMD analysis.

file_globs = [os.path.join(DATA_DIR, pat) for pat in FILE_PATTERNS]
matched_files = sorted({f for pat in file_globs for f in glob.glob(pat)})
if not matched_files:
    raise FileNotFoundError(
        'No files matched FILE_PATTERNS in DATA_DIR. '
        f'DATA_DIR={DATA_DIR}, FILE_PATTERNS={FILE_PATTERNS}'
    )

ds = xr.open_mfdataset(
    matched_files,
    chunks={'ocean_time': 1},
    data_vars='minimal',
    coords='minimal',
    compat='override',
)

os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

var_name = next((v for v in VARIABLE_CHOICES if v in ds.data_vars), None)
if var_name is None:
    raise ValueError(
        f'None of VARIABLE_CHOICES={VARIABLE_CHOICES} found in loaded data vars: '
        f'{list(ds.data_vars)}'
    )

u_da = ds[var_name]
if 'ocean_time' not in u_da.dims:
    raise ValueError(
        f"Selected variable '{var_name}' must include 'ocean_time'. "
        f'Found dims: {u_da.dims}'
    )

isel_map = {'ocean_time': slice(TIME_START, TIME_STOP, TIME_STEP)}
if 's_rho' in u_da.dims:
    isel_map['s_rho'] = -1

# Use the actual horizontal dims for u (typically eta_u/xi_u, not eta_rho/xi_rho).
y_dim = u_da.dims[-2]
x_dim = u_da.dims[-1]
isel_map[y_dim] = Y_SLICE
isel_map[x_dim] = X_SLICE

u_sub = u_da.isel(**isel_map)
U_cube = u_sub.values.astype(float)
Nt, Ny, Nx = U_cube.shape
print(f"Loaded variable '{var_name}' cube (t,y,x) = {U_cube.shape}")

# ---- Masking: identify valid (non-land) grid points ----
# ROMS data contains NaN at land points. A point is "valid" if it has
# finite values at ALL time steps (valid over entire time series).
# This ensures DMD analysis only uses consistently wet points.
mask = np.isfinite(U_cube).all(axis=0)
n_valid = int(mask.sum())
print(f'Valid (wet) points: {n_valid} / {Ny*Nx}')
if n_valid == 0:
    raise RuntimeError('No valid points in selected box.')

def cube_to_mat(cube):
    """Convert spatial-temporal cube to two-dimensional matrix for DMD analysis.
    
    Reshapes (t, y, x) array into (t, y*x) flattened form, applies spatial mask,
    then transposes to (n_valid, t) format for DMD methods.
    DMD convention: columns = time snapshots, rows = spatial grid points.
    Only valid (non-NaN, wet ocean) points are retained.
    
    Args:
        cube: array of shape (Nt, Ny, Nx)
    Returns:
        matrix: array of shape (n_valid_spatial_points, Nt)
    """
    return cube.reshape(cube.shape[0], -1)[:, mask.ravel()].T

def mat_to_cube(mat):
    """Inverse of cube_to_mat: convert matrix back to three-dimensional spatial cube.
    
    Reconstructs (n_valid, t) matrix into (t, y*x) flattened then (t, y, x) cube format.
    Non-valid masked points are filled with NaN to preserve original domain shape.
    
    Args:
        mat: array of shape (n_valid_spatial_points, Nt)
    Returns:
        cube: array of shape (Nt, Ny, Nx) with NaN at masked points
    """
    t_dim = mat.shape[1]
    out = np.full((t_dim, Ny * Nx), np.nan)
    out[:, mask.ravel()] = mat.T
    return out.reshape(t_dim, Ny, Nx)

Xmat       = cube_to_mat(U_cube)   # (n_valid, Nt) matrix of valid-only points
t          = np.arange(Nt)            # Time index array for plotting

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
pyi = PROBE_Y - y0
pxi = PROBE_X - x0
if not (0 <= pyi < Ny and 0 <= pxi < Nx):
    raise ValueError('Probe point lies outside the selected box.')
if not mask[pyi, pxi]:
    raise ValueError('Probe point is masked (land).')

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

print('  ResDMD...')
forecasts_mat['ResDMD'] = run_method_over_segments(
    resdmd_forecast_field,
    RESDMD_POD_RANK, RESDMD_DELAYS,
    RESDMD_POLY_ORDER, RESDMD_TOL, RESDMD_MIN_KEEP)

# Convert each method's (n_valid, Nt) forecast -> (Nt, Ny, Nx) cube
for name, f in forecasts_mat.items():
    forecasts_cube[name] = mat_to_cube(f)

# =====================================================================
# VISUALIZATION & ANALYSIS
# =====================================================================
# Generate 4 figures comparing forecast methods:
# 1. Probe time series at (PROBE_Y, PROBE_X) + forecast error
# 2. Spatial RMSE map per method (grid-by-grid error)
# 3. Snapshot comparison at one forecast step (spatial fields)
# 4. Per-step forecast RMSE (error evolution in time)
# Plus spatial snapshot frames (one PNG per forecast step)

# =====================================================================
# FIGURE 1 — Probe time series + error
# =====================================================================
# Top panel: velocity time series at probe location showing observed vs. all forecast methods
# Bottom panel: forecast error (predicted - observed) evolution for each method
# Red line at y=0 marks zero error. Watch how error diverges with time for each method.

probe_obs  = U_cube[:, pyi, pxi]
probe_true = U_cube[forecast_time_idx, pyi, pxi]

fig1, axes1 = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

axes1[0].plot(t, probe_obs, 'k', lw=2, label='observed')
for name, fc in forecasts_cube.items():
    axes1[0].plot(t, fc[:, pyi, pxi], '-o', lw=1.2, ms=3, label=name)
axes1[0].set_ylabel('u')
axes1[0].set_title(f'Probe (x={PROBE_X}, y={PROBE_Y}) — forecasts')
axes1[0].legend(ncol=3, fontsize=8, loc='best')

for name, fc in forecasts_cube.items():
    err_probe = fc[forecast_time_idx, pyi, pxi] - probe_true
    axes1[1].plot(forecast_time_idx, err_probe, lw=1.2, label=name)
axes1[1].axhline(0, color='k', lw=0.5)
axes1[1].set_ylabel('forecast − truth')
axes1[1].set_xlabel('time index')
axes1[1].set_title('Probe forecast error')
axes1[1].legend(ncol=3, fontsize=8, loc='best')

plt.tight_layout()
fig1.savefig(os.path.join(PLOT_OUTPUT_DIR, 'figure1_probe_timeseries_error.png'), dpi=150)

# =====================================================================
# FIGURE 2 — Spatial RMSE map per method
# =====================================================================
# =====================================================================
# FIGURE 2 — Spatial RMSE map per method
# =====================================================================
# Grid-by-grid error magnitude for each method (averaged over all forecast steps).
# Identifies regional strengths/weaknesses. Red 'X' marks probe location.
# Common color scale across all subplots aids visual comparison.
# Title shows spatial mean RMSE for quick ranking of methods.
method_names = list(forecasts_cube.keys())
nm = len(method_names)
ncols = 4
nrows = int(np.ceil(nm / ncols))

fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                           squeeze=False)

# common colour scale across methods
rmse_maps = {}
for name, fc in forecasts_cube.items():
    err = fc[forecast_time_idx] - U_cube[forecast_time_idx]
    rmse_maps[name] = np.sqrt(np.nanmean(err ** 2, axis=0))

vmax = max(np.nanmax(r) for r in rmse_maps.values())
vmin = 0.0

for k, name in enumerate(method_names):
    ax = axes2[k // ncols, k % ncols]
    im = ax.imshow(rmse_maps[name], origin='lower',
                   vmin=vmin, vmax=vmax, cmap='viridis')
    ax.set_title(f'{name}  (mean={np.nanmean(rmse_maps[name]):.3e})',
                 fontsize=9)
    ax.plot(pxi, pyi, 'rx', ms=8, mew=2)  # mark probe
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# hide unused axes
for k in range(nm, nrows * ncols):
    axes2[k // ncols, k % ncols].axis('off')

fig2.suptitle(f'Spatial RMSE (averaged over {n_fc} forecast steps)',
              fontsize=12)
plt.tight_layout()
fig2.savefig(os.path.join(PLOT_OUTPUT_DIR, 'figure2_spatial_rmse_map.png'), dpi=150)

# =====================================================================
# FIGURE 3 — Snapshot comparison at step SNAP_STEP
# =====================================================================
# =====================================================================
# FIGURE 3 — Snapshot comparison at step SNAP_STEP
# =====================================================================
# Snapshot view at a representative forecast step showing truth + all method predictions.
# Left panel: ground truth velocity field at this time
# Other panels: each method's predicted field at the same time
# Same color scale across all panels (derived from truth range)
# Red 'X' marks probe location for spatial reference
snap = int(np.clip(SNAP_STEP, 0, n_fc - 1))
snap_t = int(forecast_time_idx[snap])
truth_snap = U_cube[snap_t]

# colour range from truth
tmin = np.nanmin(truth_snap)
tmax = np.nanmax(truth_snap)

npanels = nm + 1                     # +1 for truth
ncols3  = 4
nrows3  = int(np.ceil(npanels / ncols3))

fig3, axes3 = plt.subplots(nrows3, ncols3,
                           figsize=(4 * ncols3, 3.5 * nrows3),
                           squeeze=False)

ax = axes3[0, 0]
im = ax.imshow(truth_snap, origin='lower', vmin=tmin, vmax=tmax, cmap='RdBu_r')
ax.set_title(f'Truth (forecast step {snap}, t={snap_t})', fontsize=9)
ax.plot(pxi, pyi, 'kx', ms=8, mew=2)
ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for k, name in enumerate(method_names, start=1):
    ax = axes3[k // ncols3, k % ncols3]
    im = ax.imshow(forecasts_cube[name][snap_t], origin='lower',
                   vmin=tmin, vmax=tmax, cmap='RdBu_r')
    ax.set_title(name, fontsize=9)
    ax.plot(pxi, pyi, 'kx', ms=8, mew=2)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for k in range(npanels, nrows3 * ncols3):
    axes3[k // ncols3, k % ncols3].axis('off')

fig3.suptitle(f'Forecast snapshot at step {snap} '
              f'(t index {snap_t})', fontsize=12)
plt.tight_layout()
fig3.savefig(os.path.join(PLOT_OUTPUT_DIR, 'figure3_snapshot_comparison.png'), dpi=150)

# =====================================================================
# SAVE 2D SPATIAL SNAPSHOT FRAMES  (one PNG per forecast step)
# Each frame shows truth on the left + one panel per method
# =====================================================================
# =====================================================================
# SAVE 2D SPATIAL SNAPSHOT FRAMES  (one PNG per forecast step)
# =====================================================================
# Export spatial field snapshots to create visual time-lapse comparison.
# Each frame shows: truth (left) + one panel per method (showing predicted field).
# Useful for identifying where/when each method succeeds or fails.
# Set SAVE_SPATIAL_SNAPSHOTS=False to skip (saves disk space & time).
if SAVE_SPATIAL_SNAPSHOTS:
    out_dir = os.path.join(PLOT_OUTPUT_DIR, SPATIAL_SNAPSHOT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    print(f'Saving {n_fc} spatial snapshot frames to: {out_dir}')

    truth_finite = truth_cube[np.isfinite(truth_cube)]
    if SPATIAL_COLOR_LIMIT is not None:
        vmax = float(abs(SPATIAL_COLOR_LIMIT))
    elif truth_finite.size > 0:
        vmax = float(np.percentile(np.abs(truth_finite), 99))
    else:
        vmax = 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    vmin = -vmax

    method_names_snap = list(forecasts_cube.keys())
    nm_snap = len(method_names_snap)
    ncols_s = min(4, nm_snap + 1)
    nrows_s = int(np.ceil((nm_snap + 1) / ncols_s))

    for i, tt in enumerate(forecast_time_idx):
        fig_s, axes_s = plt.subplots(
            nrows_s, ncols_s,
            figsize=(4 * ncols_s, 3.5 * nrows_s),
            squeeze=False,
        )

        ax = axes_s[0, 0]
        im = ax.imshow(U_cube[tt], origin='lower',
                       vmin=vmin, vmax=vmax, cmap=SPATIAL_CMAP)
        ax.set_title(f'Truth  (t={tt})', fontsize=9)
        ax.plot(pxi, pyi, 'kx', ms=6, mew=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for k, name in enumerate(method_names_snap, start=1):
            ax = axes_s[k // ncols_s, k % ncols_s]
            im = ax.imshow(forecasts_cube[name][tt], origin='lower',
                           vmin=vmin, vmax=vmax, cmap=SPATIAL_CMAP)
            ax.set_title(name, fontsize=9)
            ax.plot(pxi, pyi, 'kx', ms=6, mew=1.5)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for k in range(nm_snap + 1, nrows_s * ncols_s):
            axes_s[k // ncols_s, k % ncols_s].axis('off')

        fig_s.suptitle(f'Forecast step {i:05d}  (t index {tt})', fontsize=11)
        fig_s.tight_layout()
        fpath = os.path.join(out_dir, f'{i:05d}.png')
        fig_s.savefig(fpath, dpi=SPATIAL_SNAPSHOT_DPI)
        plt.close(fig_s)
        print(f'  saved frame {i + 1}/{n_fc}', end='\r')

    print(f'\nDone — {n_fc} frames saved to: {out_dir}')

# =====================================================================
# FIGURE 4 — Per-step spatial RMSE time series
# =====================================================================
# =====================================================================
# FIGURE 4 — Per-step spatial RMSE time series
# =====================================================================
# Time-resolved forecast skill: shows how error grows from step 0 (best, shortest lead time)
# to final forecast step (worst, longest lead time).
# Steep rise = method loses skill quickly
# Gentle slope = method maintains coherent predictions longer
# Useful for understanding predictability horizon and method reliability
fig4, ax4 = plt.subplots(1, 1, figsize=(11, 6))

for name, fc in forecasts_cube.items():
    # compute spatial mean RMSE at each forecasted time index
    err = fc[forecast_time_idx] - U_cube[forecast_time_idx]
    rmse_per_step = np.sqrt(np.nanmean(err ** 2, axis=(1, 2)))
    ax4.plot(range(n_fc), rmse_per_step, '-o', lw=1.5, ms=4, label=name)

ax4.set_xlabel('Forecast step')
ax4.set_ylabel('Spatial mean RMSE')
ax4.set_title('Per-step forecast RMSE (averaged over spatial domain)')
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3)
fig4.savefig(os.path.join(PLOT_OUTPUT_DIR, 'figure4_forecast_rmse_per_step.png'), dpi=150)
plt.tight_layout()

# =====================================================================
# RMSE REPORT
# =====================================================================
# =====================================================================
# RMSE REPORT
# =====================================================================
# Print summary statistics comparing forecast skill across all methods.
# - Field RMSE: error averaged over space (all wet points) and time (all forecast steps)
# - Probe RMSE: error at single grid point (PROBE_X, PROBE_Y) only
# Lower RMSE is better. Per-step table shows how error grows with forecast lead time.
print('\nForecast RMSE (space+time averaged over wet points):')
for name, fc in forecasts_cube.items():
    err = fc - U_cube
    rmse_full  = float(np.sqrt(np.nanmean(err ** 2)))
    rmse_probe = float(np.sqrt(np.nanmean(
        (fc[:, pyi, pxi] - U_cube[:, pyi, pxi]) ** 2)))
    print(f'  {name:16s}  field RMSE = {rmse_full:.4e}   '
          f'probe RMSE = {rmse_probe:.4e}')

# Per-step (time-resolved) field RMSE, optional summary
print('\nPer-step field RMSE (first 5 forecast steps):')
header = '  step  ' + '  '.join(f'{n:>14s}' for n in forecasts_cube.keys())
print(header)
for i in range(min(5, n_fc)):
    tt = int(forecast_time_idx[i])
    row = f'  {i:4d} (t={tt:4d})  '
    for name, fc in forecasts_cube.items():
        e = fc[tt] - U_cube[tt]
        row += f'  {np.sqrt(np.nanmean(e**2)):14.4e}'
    print(row)

print('\nDone.')