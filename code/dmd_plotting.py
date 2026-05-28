"""Visualization and reporting functions for DMD forecast comparison.

All functions accept explicit array arguments and produce self-contained
figures. They write nothing to disk unless given an output path.

Functions
---------
plot_probe_timeseries       — Fig 1: time series at diagnostic probe points
plot_spatial_rmse_map       — Fig 2: grid-by-grid RMSE map per method
plot_snapshot_comparison    — Fig 3: spatial field comparison at one step
save_spatial_snapshot_frames — export one PNG per forecast step
plot_forecast_rmse_per_step — Fig 4: per-step spatial mean RMSE curves
print_rmse_report           — console RMSE summary table
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from cmocean import cm as cmo

# Default colormap used when the caller does not specify one
_DEFAULT_CMAP = cmo.balance


# =====================================================================
# FIGURE 1 — Probe time series
# =====================================================================

def plot_probe_timeseries(probe_points, t, truth_cube, forecasts_cube,
                          output_path, var_label='u'):
    """Plot observed and forecast time series at each diagnostic probe point.

    Parameters
    ----------
    probe_points   : list of probe dicts (from dmd_utils.build_probe_list)
    t              : 1-D array — time index
    truth_cube     : ndarray, shape (Nt, Ny, Nx) — ground-truth field
    forecasts_cube : dict {method_name: ndarray (Nt, Ny, Nx)}
    output_path    : str — full file path for the saved figure
    var_label      : str — variable name for axis labels
    """
    n_probe = len(probe_points)
    fig_h = max(3.0 * n_probe, 6.0)
    fig, axes = plt.subplots(n_probe, 1, figsize=(12, fig_h),
                             sharex=True, squeeze=False)
    axes = axes.ravel()

    for ip, p in enumerate(probe_points):
        ax = axes[ip]
        pxi, pyi = p['pxi'], p['pyi']
        ax.plot(t, truth_cube[:, pyi, pxi], 'k', lw=2, label='observed')
        for name, fc in forecasts_cube.items():
            ax.plot(t, fc[:, pyi, pxi], lw=1.2, label=name)
        ax.set_ylabel(var_label)
        ax.set_title(f"{p['name']} (x={p['x_full']}, y={p['y_full']})")
        if ip == 0:
            ax.legend(ncol=4, fontsize=8, loc='best')

    axes[-1].set_xlabel('time index')
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# =====================================================================
# FIGURE 2 — Spatial RMSE map
# =====================================================================

def plot_spatial_rmse_map(forecasts_cube, truth_cube, forecast_time_idx,
                          probe_points, n_fc, output_path, title_suffix=''):
    """Plot grid-by-grid RMSE maps for every forecast method.

    Parameters
    ----------
    forecasts_cube    : dict {method_name: ndarray (Nt, Ny, Nx)}
    truth_cube        : ndarray, shape (Nt, Ny, Nx)
    forecast_time_idx : 1-D int array — time indices of forecast steps
    probe_points      : list of probe dicts
    n_fc              : int — total number of forecast steps
    output_path       : str
    title_suffix      : str — appended to the figure title (e.g. variable name)
    """
    method_names = list(forecasts_cube.keys())
    nm = len(method_names)
    ncols = 4
    nrows = int(np.ceil(nm / ncols))

    rmse_maps = {}
    for name, fc in forecasts_cube.items():
        err = fc[forecast_time_idx] - truth_cube[forecast_time_idx]
        rmse_maps[name] = np.sqrt(np.nanmean(err ** 2, axis=0))

    vmax = max(np.nanmax(r) for r in rmse_maps.values())

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for k, name in enumerate(method_names):
        ax = axes[k // ncols, k % ncols]
        im = ax.imshow(rmse_maps[name], origin='lower',
                       vmin=0.0, vmax=vmax, cmap='viridis')
        ax.set_title(f'{name}  (mean={np.nanmean(rmse_maps[name]):.3e})',
                     fontsize=9)
        for p in probe_points:
            ax.plot(p['pxi'], p['pyi'], 'wx', ms=6, mew=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for k in range(nm, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')

    suffix = f' — {title_suffix}' if title_suffix else ''
    fig.suptitle(
        f'Spatial RMSE (averaged over {n_fc} forecast steps){suffix}',
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# =====================================================================
# FIGURE 3 — Snapshot comparison
# =====================================================================

def plot_snapshot_comparison(forecasts_cube, truth_cube, forecast_time_idx,
                             probe_points, snap_step, output_path,
                             cmap=None, vmin=None, vmax=None,
                             title_suffix=''):
    """Side-by-side spatial field comparison at a single forecast step.

    Parameters
    ----------
    forecasts_cube    : dict {method_name: ndarray (Nt, Ny, Nx)}
    truth_cube        : ndarray, shape (Nt, Ny, Nx)
    forecast_time_idx : 1-D int array
    probe_points      : list of probe dicts
    snap_step         : int — which forecast step index to visualise
    output_path       : str
    cmap              : matplotlib colormap or None (defaults to balance)
    vmin, vmax        : float or None — colour limits (auto-derived if None)
    title_suffix      : str
    """
    if cmap is None:
        cmap = _DEFAULT_CMAP

    snap = int(np.clip(snap_step, 0, len(forecast_time_idx) - 1))
    snap_t = int(forecast_time_idx[snap])
    truth_snap = truth_cube[snap_t]

    if vmin is None or vmax is None:
        tmin = float(np.nanmin(truth_snap))
        tmax = float(np.nanmax(truth_snap))
        vmin = vmin if vmin is not None else tmin
        vmax = vmax if vmax is not None else tmax

    method_names = list(forecasts_cube.keys())
    npanels = len(method_names) + 1
    ncols = 4
    nrows = int(np.ceil(npanels / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    ax = axes[0, 0]
    im = ax.imshow(truth_snap, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(f'Truth (step {snap}, t={snap_t})', fontsize=9)
    for p in probe_points:
        ax.plot(p['pxi'], p['pyi'], 'kx', ms=6, mew=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for k, name in enumerate(method_names, start=1):
        ax = axes[k // ncols, k % ncols]
        im = ax.imshow(forecasts_cube[name][snap_t], origin='lower',
                       vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(name, fontsize=9)
        for p in probe_points:
            ax.plot(p['pxi'], p['pyi'], 'kx', ms=6, mew=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for k in range(npanels, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')

    suffix = f' — {title_suffix}' if title_suffix else ''
    fig.suptitle(f'Forecast snapshot at step {snap} (t={snap_t}){suffix}',
                 fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# =====================================================================
# SPATIAL SNAPSHOT FRAMES
# =====================================================================

def save_spatial_snapshot_frames(forecasts_cube, truth_cube, forecast_time_idx,
                                 probe_points, out_dir,
                                 cmap=None, vmin=None, vmax=None,
                                 dpi=120, var_label=''):
    """Export one PNG per forecast step showing truth and all method fields.

    Parameters
    ----------
    forecasts_cube    : dict {method_name: ndarray (Nt, Ny, Nx)}
    truth_cube        : ndarray, shape (Nt, Ny, Nx)
    forecast_time_idx : 1-D int array
    probe_points      : list of probe dicts
    out_dir           : str — directory for output frames
    cmap              : matplotlib colormap or None (defaults to balance)
    vmin, vmax        : float or None — colour limits (auto-derived from truth if None)
    dpi               : int
    var_label         : str — used in frame title and progress messages
    """
    if cmap is None:
        cmap = _DEFAULT_CMAP

    os.makedirs(out_dir, exist_ok=True)
    n_fc = len(forecast_time_idx)

    if vmin is None or vmax is None:
        truth_finite = truth_cube[np.isfinite(truth_cube)]
        auto_vmax = (float(np.percentile(np.abs(truth_finite), 99))
                     if truth_finite.size > 0 else 1.0)
        if not np.isfinite(auto_vmax) or auto_vmax <= 0:
            auto_vmax = 1.0
        vmin = vmin if vmin is not None else -auto_vmax
        vmax = vmax if vmax is not None else auto_vmax

    method_names = list(forecasts_cube.keys())
    nm = len(method_names)
    ncols_s = min(4, nm + 1)
    nrows_s = int(np.ceil((nm + 1) / ncols_s))

    label_prefix = f'[{var_label}] ' if var_label else ''
    print(f'{label_prefix}Saving {n_fc} spatial snapshot frames to: {out_dir}')

    for i, tt in enumerate(forecast_time_idx):
        fig_s, axes_s = plt.subplots(
            nrows_s, ncols_s,
            figsize=(4 * ncols_s, 3.5 * nrows_s),
            squeeze=False,
        )

        ax = axes_s[0, 0]
        im = ax.imshow(truth_cube[tt], origin='lower',
                       vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(f'Truth  (t={tt})', fontsize=9)
        for p in probe_points:
            ax.plot(p['pxi'], p['pyi'], 'kx', ms=5, mew=1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for k, name in enumerate(method_names, start=1):
            ax = axes_s[k // ncols_s, k % ncols_s]
            im = ax.imshow(forecasts_cube[name][tt], origin='lower',
                           vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_title(name, fontsize=9)
            for p in probe_points:
                ax.plot(p['pxi'], p['pyi'], 'kx', ms=5, mew=1.2)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for k in range(nm + 1, nrows_s * ncols_s):
            axes_s[k // ncols_s, k % ncols_s].axis('off')

        title = f'{var_label + " — " if var_label else ""}Forecast step {i:05d}  (t={tt})'
        fig_s.suptitle(title, fontsize=11)
        fig_s.tight_layout()
        fig_s.savefig(os.path.join(out_dir, f'{i:05d}.png'), dpi=dpi)
        plt.close(fig_s)
        print(f'  {label_prefix}saved frame {i + 1}/{n_fc}', end='\r')

    print(f'\n{label_prefix}Done — {n_fc} frames saved to: {out_dir}')


# =====================================================================
# FIGURE 4 — Per-step spatial RMSE
# =====================================================================

def plot_forecast_rmse_per_step(forecasts_cubes, truth_cubes,
                                forecast_time_idx, n_fc, output_path,
                                var_labels=None):
    """Plot per-step spatial mean RMSE, with one subplot per variable.

    Accepts either a single pair (univar) or lists of pairs (multivar).

    Parameters
    ----------
    forecasts_cubes   : dict or list of dicts {method_name: (Nt, Ny, Nx)}
        Single dict for univar; list of dicts (one per variable) for multivar.
    truth_cubes       : ndarray or list of ndarrays, shape (Nt, Ny, Nx)
    forecast_time_idx : 1-D int array
    n_fc              : int
    output_path       : str
    var_labels        : str or list of str or None — subplot titles
    """
    # Normalise to lists so the loop below handles both univar and multivar
    if isinstance(forecasts_cubes, dict):
        forecasts_cubes = [forecasts_cubes]
        truth_cubes = [truth_cubes]
        var_labels = [var_labels] if isinstance(var_labels, str) else [var_labels or '']
    else:
        if var_labels is None:
            var_labels = ['' for _ in forecasts_cubes]

    nvars = len(forecasts_cubes)
    fig, axes = plt.subplots(nvars, 1,
                             figsize=(11, 4 * nvars),
                             sharex=True, squeeze=False)

    for iv, (fc_dict, truth_cube, vl) in enumerate(
            zip(forecasts_cubes, truth_cubes, var_labels)):
        ax = axes[iv, 0]
        for name, fc in fc_dict.items():
            err = fc[forecast_time_idx] - truth_cube[forecast_time_idx]
            rmse_per_step = np.sqrt(np.nanmean(err ** 2, axis=(1, 2)))
            ax.plot(range(n_fc), rmse_per_step, '-o', lw=1.5, ms=4, label=name)
        ylabel = f'RMSE ({vl})' if vl else 'Spatial mean RMSE'
        ax.set_ylabel(ylabel)
        title = f'{vl} — per-step spatial mean RMSE' if vl else 'Per-step spatial mean RMSE'
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Forecast step')
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# =====================================================================
# RMSE REPORT
# =====================================================================

def print_rmse_report(forecasts_cube, truth_cube, probe_points,
                      forecast_time_idx, n_fc, var_name=''):
    """Print field-averaged and probe-averaged RMSE for every method.

    Also prints a per-step RMSE table for the first 5 forecast steps.

    Parameters
    ----------
    forecasts_cube    : dict {method_name: ndarray (Nt, Ny, Nx)}
    truth_cube        : ndarray, shape (Nt, Ny, Nx)
    probe_points      : list of probe dicts
    forecast_time_idx : 1-D int array
    n_fc              : int
    var_name          : str — included in the header if provided
    """
    header = f'Forecast RMSE — {var_name}' if var_name else 'Forecast RMSE'
    print(f'\n{header} (space+time averaged over wet points):')

    for name, fc in forecasts_cube.items():
        err = fc - truth_cube
        rmse_full = float(np.sqrt(np.nanmean(err ** 2)))
        probe_sqerr = []
        for p in probe_points:
            probe_sqerr.append(
                (fc[:, p['pyi'], p['pxi']] - truth_cube[:, p['pyi'], p['pxi']]) ** 2
            )
        rmse_probe = float(np.sqrt(np.nanmean(np.stack(probe_sqerr, axis=0))))
        print(f'  {name:16s}  field RMSE = {rmse_full:.4e}   '
              f'probe-set RMSE = {rmse_probe:.4e}')

    print(f'\nPer-step field RMSE — first 5 steps:')
    header_row = '  step  ' + '  '.join(f'{n:>14s}' for n in forecasts_cube.keys())
    print(header_row)
    for i in range(min(5, n_fc)):
        tt = int(forecast_time_idx[i])
        row = f'  {i:4d} (t={tt:4d})  '
        for name, fc in forecasts_cube.items():
            e = fc[tt] - truth_cube[tt]
            row += f'  {np.sqrt(np.nanmean(e ** 2)):14.4e}'
        print(row)
