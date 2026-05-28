"""Pure DMD algorithm implementations.

All functions are stateless and operate only on numpy arrays passed as arguments.
No global state, no I/O. Import this module from runner scripts.

METHODS:
  Helpers:
    build_hankel_mat      — time-delay (Hankel) embedding
    dmd_fit               — core SVD-based DMD fit
    poly_dict             — polynomial feature dictionary for EDMD/ResDMD
    rbf_gram_cols         — Gaussian RBF Gram matrix
    generalized_eig_stable — regularized generalized eigenproblem

  Forecast functions (all return (n_features, n_fc) arrays):
    dmd_forecast_field
    sliding_dmd_forecast_field
    hankel_dmd_forecast_field
    sliding_hankel_dmd_forecast_field
    edmd_forecast_field
    kernel_dmd_forecast_field
    dmd_matlab_forecast_field
    kedmd_forecast_field
    specrkhs_obs_forecast_field
    resdmd_forecast_field
"""

import numpy as np
from numpy.linalg import pinv, svd, eig


# =====================================================================
# LOW-LEVEL HELPERS
# =====================================================================

def build_hankel_mat(X, d):
    """Construct Hankel (time-delay embedding) matrix from data.

    Stacks d consecutive snapshots as rows:
        [X[:, 0:n], X[:, 1:n+1], ..., X[:, d-1:n+d-1]]

    Parameters
    ----------
    X : ndarray, shape (n_features, m_samples)
    d : int
        Number of delays.

    Returns
    -------
    H : ndarray, shape (d * n_features, m_samples - d + 1)
    """
    n = X.shape[1] - d + 1
    return np.vstack([X[:, i:i + n] for i in range(d)])


def dmd_fit(X, Y, r=None):
    """Fit DMD Koopman matrix from snapshot pairs X (input) and Y (output).

    Solves A = Y @ X^+ in the POD-r subspace.

    Parameters
    ----------
    X, Y : ndarray, shape (n_features, m_snapshots)
    r    : int or None
        SVD truncation rank. Uses all non-zero singular values if None.

    Returns
    -------
    A_t : ndarray, shape (r, r)  — DMD matrix in POD space
    U_r : ndarray, shape (n_features, r) — POD basis (left singular vectors)
    """
    U, S, Vt = svd(X, full_matrices=False)
    if r is None or r > len(S):
        r = len(S)
    U_r = U[:, :r]
    S_r = S[:r]
    V_r = Vt[:r, :].conj().T
    A_t = U_r.conj().T @ Y @ V_r / S_r
    return A_t, U_r


def poly_dict(x, order):
    """Build polynomial dictionary for EDMD/ResDMD.

    Returns stacked feature rows: [1, x, x^2, ..., x^order].

    Parameters
    ----------
    x     : ndarray, shape (n_features, m_samples)
    order : int — polynomial order

    Returns
    -------
    feats : ndarray, shape (1 + n_features * order, m_samples)
    """
    feats = [np.ones((1, x.shape[1]))]
    for p in range(1, order + 1):
        feats.append(x ** p)
    return np.vstack(feats)


def rbf_gram_cols(Xa, Xb, sigma=None):
    """Gaussian RBF Gram matrix between columns of Xa and Xb.

    Parameters
    ----------
    Xa     : ndarray, shape (n_features, n_a)
    Xb     : ndarray, shape (n_features, n_b)
    sigma  : float or None — RBF bandwidth (auto from median pairwise distance if None)

    Returns
    -------
    K      : ndarray, shape (n_a, n_b)
    sigma  : float — bandwidth actually used
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
    """Solve generalized eigenvalue problem A v = lambda G v via regularized solve.

    Parameters
    ----------
    A, G : ndarray, shape (n, n)
    reg  : float — Tikhonov regularization added to G diagonal

    Returns
    -------
    vals : ndarray, shape (n,)  — eigenvalues
    vecs : ndarray, shape (n, n) — eigenvectors (columns)
    """
    n = G.shape[0]
    G_reg = G + reg * np.eye(n)
    M = np.linalg.solve(G_reg, A)
    vals, vecs = eig(M)
    return vals, vecs


# =====================================================================
# FORECAST METHODS
# =====================================================================
# All methods accept:
#   Xtr  : ndarray, shape (n_features, n_train) — training snapshots
#   n_fc : int — number of forecast steps
#   ...additional hyperparameters...
# All methods return:
#   out  : ndarray, shape (n_features, n_fc) — forecast snapshots

def dmd_forecast_field(Xtr, n_fc, r=None):
    """Standard DMD: fit once, iterate Koopman operator forward.

    Parameters
    ----------
    Xtr  : ndarray, shape (n_features, n_train)
    n_fc : int
    r    : int or None — SVD truncation rank
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
    """Sliding-window DMD: re-fit every ``stride`` steps on the most recent ``window`` snapshots.

    Parameters
    ----------
    Xtr    : ndarray, shape (n_features, n_train)
    n_fc   : int
    window : int — number of recent snapshots used for each fit
    r      : int or None — SVD truncation rank
    stride : int — how often to re-fit (1 = every step)
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
    """Hankel DMD: time-delay embedding to expose hidden linear structure.

    Parameters
    ----------
    Xtr  : ndarray, shape (n_features, n_train)
    n_fc : int
    d    : int — number of delays
    r    : int or None — SVD truncation rank
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
    """Sliding Hankel DMD: combines sliding window re-fitting with delay embedding.

    Parameters
    ----------
    Xtr    : ndarray, shape (n_features, n_train)
    n_fc   : int
    window : int — recent snapshots used per fit
    d      : int — number of delays in Hankel embedding
    r      : int or None — SVD truncation rank
    stride : int — how often to re-fit
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
    """Extended DMD (EDMD): POD + Hankel delay embedding + polynomial Koopman dictionary.

    Parameters
    ----------
    Xtr      : ndarray, shape (n_features, n_train)
    n_fc     : int
    pod_rank : int — POD truncation before delay embedding
    d        : int — number of delays
    order    : int — polynomial dictionary order
    """
    U, S, Vt = svd(Xtr, full_matrices=False)
    r = min(pod_rank, len(S))
    U_r = U[:, :r]
    A_coef = U_r.T @ Xtr
    scale = np.max(np.abs(A_coef)) + 1e-12
    A_s = A_coef / scale
    H = build_hankel_mat(A_s, d)
    Phi_X = poly_dict(H[:, :-1], order)
    Phi_Y = poly_dict(H[:, 1:], order)
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
    """Kernel DMD: Gaussian RBF kernel on delay-embedded snapshots.

    Parameters
    ----------
    Xtr   : ndarray, shape (n_features, n_train)
    n_fc  : int
    d     : int — delay dimension for Hankel embedding
    sigma : float or None — RBF bandwidth (auto-computed if None)
    r     : int — rank in kernel feature space
    reg   : float — Tikhonov regularization for decoder
    """
    H = build_hankel_mat(Xtr, d)
    nf = Xtr.shape[0]
    X = H[:, :-1].T   # (m, d*nf)
    Y = H[:, 1:].T

    def sqdist(A, B):
        return (np.sum(A ** 2, 1)[:, None]
                + np.sum(B ** 2, 1)[None, :]
                - 2 * A @ B.T)

    D2 = sqdist(X, X)
    if sigma is None:
        sigma = np.sqrt(np.median(D2[D2 > 0]) + 1e-12)
    Gxx = np.exp(-D2 / (2 * sigma ** 2))
    Gyx = np.exp(-sqdist(Y, X) / (2 * sigma ** 2))

    evals, Q = np.linalg.eigh(Gxx)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    Q = Q[:, idx]
    rr = int(min(r, np.sum(evals > 1e-10)))
    evals = evals[:rr]
    Q = Q[:, :rr]
    Sig = np.sqrt(np.clip(evals, 1e-12, None))
    Sinv = np.diag(1.0 / Sig)

    K_hat = Sinv @ Q.T @ Gyx @ Q @ Sinv

    g0 = np.exp(-sqdist(X[-1:], X) / (2 * sigma ** 2))[0]
    z = Sinv @ Q.T @ g0

    Z = Sinv @ Q.T @ Gxx
    C = X.T @ Z.T @ np.linalg.inv(Z @ Z.T + reg * np.eye(rr))

    out = np.zeros((nf, n_fc))
    for i in range(n_fc):
        z = K_hat @ z
        x_full = np.real(C @ z)
        out[:, i] = x_full[-nf:]
    return out


def dmd_matlab_forecast_field(Xtr, n_fc):
    """DMD forecast mirroring the MATLAB antarctic/sealevel workflow.

    Uses reduced-coordinate SVD, eigendecomposition of the reduced map,
    and modal reconstruction in the same style as the referenced MATLAB scripts.

    Parameters
    ----------
    Xtr  : ndarray, shape (n_features, n_train)
    n_fc : int
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
    """kEDMD forecast mirroring the MATLAB eig(A, G) Koopman mode workflow.

    Parameters
    ----------
    Xtr   : ndarray, shape (n_features, n_train)
    n_fc  : int
    sigma : float or None — RBF bandwidth
    reg   : float — generalized eigenvalue regularization
    """
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
      1. Build kernel matrices G, A, R from snapshot pairs.
      2. Compute generalized Koopman eigenpairs.
      3. Filter by residuals (SpecRKHS-inspired).
      4. Evolve observable expansion from the last training snapshot.

    Parameters
    ----------
    Xtr      : ndarray, shape (n_features, n_train)
    n_fc     : int
    sigma    : float or None — RBF bandwidth
    reg      : float — generalized eigenvalue regularization
    res_tol  : float — residual threshold for eigenpair filtering
    min_keep : int — minimum eigenpairs to retain regardless of residual
    """
    x = Xtr[:, :-1]
    y = Xtr[:, 1:]
    x0 = Xtr[:, -1]
    nf = Xtr.shape[0]

    G, sigma_used = rbf_gram_cols(x, x, sigma)
    A, _ = rbf_gram_cols(y, x, sigma_used)
    R, _ = rbf_gram_cols(y, y, sigma_used)

    lam, F = generalized_eig_stable(A, G, reg=reg)

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

    Steps:
      1. POD reduction to ``pod_rank`` basis vectors.
      2. Delay-embed and build polynomial Koopman operator (like EDMD).
      3. Compute residual error for each Koopman eigenpair.
      4. Keep only low-residual pairs.
      5. Forecast using filtered spectral expansion.

    Parameters
    ----------
    Xtr      : ndarray, shape (n_features, n_train)
    n_fc     : int
    pod_rank : int — POD truncation rank
    d        : int — number of delays in Hankel embedding
    order    : int — polynomial dictionary order
    tol      : float — residual threshold for eigenpair filtering
    min_keep : int — minimum eigenpairs to retain
    """
    U, S, Vt = svd(Xtr, full_matrices=False)
    r = min(pod_rank, len(S))
    U_r = U[:, :r]
    A_coef = U_r.T @ Xtr
    scale = np.max(np.abs(A_coef)) + 1e-12
    A_s = A_coef / scale

    H = build_hankel_mat(A_s, d)
    Phi_X = poly_dict(H[:, :-1], order)
    Phi_Y = poly_dict(H[:, 1:], order)
    m = Phi_X.shape[1]

    G = (Phi_X @ Phi_X.conj().T) / m
    A = (Phi_X @ Phi_Y.conj().T) / m
    L = (Phi_Y @ Phi_Y.conj().T) / m

    K = Phi_Y @ pinv(Phi_X)
    evals, V = eig(K)

    res = np.zeros(len(evals))
    for j, lam in enumerate(evals):
        v = V[:, j]
        M1 = L - np.conj(lam) * A.conj().T - lam * A + (abs(lam) ** 2) * G
        num = np.real(v.conj() @ M1 @ v)
        den = np.real(v.conj() @ G @ v)
        res[j] = np.sqrt(max(num, 0.0) / max(den, 1e-30))

    keep = np.where(res < tol)[0]
    if keep.size < min_keep:
        keep = np.argsort(res)[:min_keep]
    lam_k = evals[keep]
    V_k = V[:, keep]

    B = H[:, :-1] @ pinv(Phi_X)
    phi0 = poly_dict(H[:, -1:], order)[:, 0]
    c0, *_ = np.linalg.lstsq(V_k, phi0, rcond=None)

    nf = Xtr.shape[0]
    out = np.zeros((nf, n_fc))
    c = c0.astype(complex)
    for i in range(n_fc):
        c = lam_k * c
        phi = V_k @ c
        z_full = np.real(B @ phi)
        coef_now = z_full[-r:] * scale
        out[:, i] = U_r @ coef_now

    print(f'[ResDMD] kept {len(keep)} / {len(evals)} eigenpairs '
          f'(min res = {res.min():.2e}, max kept = {res[keep].max():.2e})')
    return out
