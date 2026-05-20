import xarray as xr
import matplotlib.pyplot as plt
import os 

# ==============================
# User parameters (edit here)
# ==============================
ROMS_FILE = "/compass/ber200006/seahorce/roms_us_east_coast/roms_his_mar.nc"
REMORA_FILE = "/compass/ber200006/seahorce/remora_us_east_coast/1km/mar.nc"

# Grid-index subset bounds (Python slicing: start inclusive, stop exclusive).
# Example: XI_START=100, XI_STOP=300 selects xi indices [100, 299].
XI_START = 750
XI_STOP = 950
ETA_START = 450
ETA_STOP = 550

# Save paths (set to None to skip writing files)
OUTDIR = '/compass/ber200006/seahorce/dmd_testing/data'


def _find_horizontal_dims(ds: xr.Dataset) -> tuple[str, str]:
	if "eta_rho" in ds.dims and "xi_rho" in ds.dims:
		return "eta_rho", "xi_rho"

	eta_candidates = [d for d in ds.dims if d.startswith("eta")]
	xi_candidates = [d for d in ds.dims if d.startswith("xi")]
	if eta_candidates and xi_candidates:
		return eta_candidates[0], xi_candidates[0]

	raise ValueError(f"Could not determine horizontal dims from dataset dims: {list(ds.dims)}")


def _subset_surface_xy(ds: xr.Dataset, xi_start: int, xi_stop: int, eta_start: int, eta_stop: int) -> xr.Dataset:
	# Subset all horizontal staggered-grid dimensions (rho/u/v/psi) consistently.
	horizontal_indexers: dict[str, slice] = {}
	for dim in ds.dims:
		if dim.startswith("eta"):
			horizontal_indexers[dim] = slice(eta_start, eta_stop)
		elif dim.startswith("xi"):
			horizontal_indexers[dim] = slice(xi_start, xi_stop)

	if not horizontal_indexers:
		eta_dim, xi_dim = _find_horizontal_dims(ds)
		horizontal_indexers = {
			eta_dim: slice(eta_start, eta_stop),
			xi_dim: slice(xi_start, xi_stop),
		}

	print(f"Using horizontal slices: {horizontal_indexers}")
	subset = ds.isel(horizontal_indexers)

	# Keep only surface for common vertical coordinates.
	if "s_rho" in subset.dims:
		subset = subset.isel(s_rho=-1)
	elif "s_w" in subset.dims:
		subset = subset.isel(s_w=-1)
	elif "z_rho" in subset.dims:
		subset = subset.isel(z_rho=-1)
	elif "z_w" in subset.dims:
		subset = subset.isel(z_w=-1)

	return subset

##### CODE ##### 
roms_ds = xr.open_dataset(ROMS_FILE)
remora_ds = xr.open_dataset(REMORA_FILE)

roms_subset = _subset_surface_xy(roms_ds, XI_START, XI_STOP, ETA_START, ETA_STOP)
remora_subset = _subset_surface_xy(remora_ds, XI_START, XI_STOP, ETA_START, ETA_STOP)

print("ROMS subset dims:", dict(roms_subset.dims))
print("REMORA subset dims:", dict(remora_subset.dims))

## delete unnecessary variables to save memory
roms_subset = roms_subset.drop_vars(['sustr', 'svstr', 'bustr', 'bvstr', 'mask_rho', 'mask_u', 'mask_v', 'mask_psi', 'ubar', 'vbar', 'omega'])
roms_subset = roms_subset.drop_vars(['ntimes', 'ndtfast', 'dt', 'dtfast', 'dstart', 'nHIS', 'ndefHIS', 'nRST', 'Falpha', 'Fbeta', 'Fgamma'])
roms_subset = roms_subset.drop_vars(['nl_tnu2', 'nl_visc2', 'LuvSponge', 'LtracerSponge', 'Akt_bak', 'Akv_bak', 'Akk_bak', 'Akp_bak'])
roms_subset = roms_subset.drop_vars(['rdrg', 'rdrg2', 'Zob', 'Zos', 'gls_p', 'gls_m', 'gls_n', 'gls_cmu0', 'gls_c1', 'gls_c2', 'gls_c3m', 'gls_c3p', 'gls_sigk', 'gls_sigp', 'gls_Kmin', 'gls_Pmin'])
roms_subset = roms_subset.drop_vars(['Charnok_alpha', 'Zos_hsig_alpha', 'sz_alpha', 'CrgBan_cw', 'Znudg', 'M2nudg', 'M3nudg', 'Tnudg', 'rho0', 'gamma2', 'LuvSrc', 'LwSrc', 'LtracerSrc'])
roms_subset = roms_subset.drop_vars(['LsshCLM', 'Lm2CLM', 'Lm3CLM', 'LtracerCLM', 'LnudgeM2CLM', 'LnudgeM3CLM', 'LnudgeTCLM', 'spherical', 'xl', 'el'])
roms_subset = roms_subset.drop_vars(['Vtransform', 'Vstretching', 'theta_s', 'theta_b', 'Tcline', 'hc', 'grid', 'Cs_r', 'Cs_w', 'h'])
roms_subset = roms_subset.drop_vars(['w'])

remora_subset = remora_subset.drop_vars(['probLo', 'probHi', 'Geom.smallend', 'Geom.bigend', 'CellSize', 'theta_s', 'theta_b', 'hc', 'grid', 'x_psi', 'y_psi'])
remora_subset = remora_subset.drop_vars(['ubar', 'vbar', 'sustr', 'svstr', 'Cs_r', 'Cs_w', 'h'])

for v in remora_subset.data_vars:
    if remora_subset[v].dtype == "float64":
        remora_subset[v] = remora_subset[v].astype("float32")

### save by variable and simulation 
## remora 
remora_temp = remora_subset.copy()
remora_temp = remora_temp.drop_vars(['pm', 'pn', 'f', 'zeta', 'salt', 'u', 'v'])
remora_temp = remora_temp.drop_vars(['s_rho', 's_w', 'x_u', 'y_v', 'y_u', 'x_v'])
remora_temp.to_netcdf(os.path.join(OUTDIR, "remora", "temp.nc"))

remora_salt = remora_subset.copy()
remora_salt = remora_salt.drop_vars([ 'pn', 'pm', 'f', 'zeta', 'u', 'v', 'temp'])
remora_salt = remora_salt.drop_vars(['s_rho', 's_w', 'x_u', 'y_v', 'y_u', 'x_v'])
remora_salt.to_netcdf(os.path.join(OUTDIR, "remora", "salt.nc"))

remora_u = remora_subset.copy()
remora_u = remora_u.drop_vars([ 'pn', 'pm', 'f', 'zeta', 'salt', 'v', 'temp'])
remora_u = remora_u.drop_vars(['s_rho', 's_w', 'x_rho', 'y_v', 'y_rho', 'x_v'])
remora_u.to_netcdf(os.path.join(OUTDIR, "remora", "u.nc"))

remora_v = remora_subset.copy()
remora_v = remora_v.drop_vars([ 'pn', 'pm', 'f', 'zeta', 'salt', 'u', 'temp'])
remora_v = remora_v.drop_vars(['s_rho', 's_w', 'x_rho', 'y_u', 'y_rho', 'x_u'])
remora_v.to_netcdf(os.path.join(OUTDIR, "remora", "v.nc"))

remora_zeta = remora_subset.copy()
remora_zeta = remora_zeta.drop_vars([ 'pn', 'pm', 'f', 'salt', 'u', 'v', 'temp'])
remora_zeta = remora_zeta.drop_vars(['s_rho', 's_w', 'x_u', 'y_v', 'y_u', 'x_v'])
remora_zeta.to_netcdf(os.path.join(OUTDIR, "remora", "zeta.nc"))

remora_pm = remora_subset.copy()
remora_pm = remora_pm.drop_vars([ 'pn', 'f', 'zeta', 'salt', 'u', 'v', 'temp'])
remora_pm = remora_pm.drop_vars(['s_rho', 's_w', 'x_u', 'y_v', 'y_u', 'x_v'])
remora_pm.to_netcdf(os.path.join(OUTDIR, "remora", "pm.nc"))

remora_pn = remora_subset.copy()
remora_pn = remora_pn.drop_vars([ 'pm', 'f', 'zeta', 'salt', 'u', 'v', 'temp'])
remora_pn = remora_pn.drop_vars(['s_rho', 's_w', 'x_u', 'y_v', 'y_u', 'x_v'])
remora_pn.to_netcdf(os.path.join(OUTDIR, "remora", "pn.nc"))

remora_f = remora_subset.copy()
remora_f = remora_f.drop_vars([ 'pn', 'pm', 'zeta', 'salt', 'u', 'v', 'temp'])
remora_f = remora_f.drop_vars(['s_rho', 's_w', 'x_u', 'y_v', 'y_u', 'x_v'])
remora_f.to_netcdf(os.path.join(OUTDIR, "remora", "f.nc"))

## roms
roms_temp = roms_subset.copy()
roms_temp = roms_temp.drop_vars(['salt', 'u', 'v', 'zeta', 'pm', 'pn', 'f'])
roms_temp = roms_temp.drop_vars(['s_rho', 's_w', 'x_u', 'y_v', 'y_u', 'x_v', 'x_psi', 'y_psi'])
roms_temp.to_netcdf(os.path.join(OUTDIR, "roms", "temp.nc"))

roms_salt = roms_subset.copy()
roms_salt = roms_salt.drop_vars(['temp', 'u', 'v', 'zeta', 'pm', 'pn', 'f'])
roms_salt = roms_salt.drop_vars(['s_rho', 's_w', 'x_u', 'y_v', 'y_u', 'x_v', 'x_psi', 'y_psi'])
roms_salt.to_netcdf(os.path.join(OUTDIR, "roms", "salt.nc"))

roms_u = roms_subset.copy()
roms_u = roms_u.drop_vars(['temp', 'salt', 'v', 'zeta', 'pm', 'pn', 'f'])
roms_u = roms_u.drop_vars(['s_rho', 's_w', 'x_rho', 'y_v', 'y_rho', 'x_v', 'x_psi', 'y_psi'])
roms_u.to_netcdf(os.path.join(OUTDIR, "roms", "u.nc"))

roms_v = roms_subset.copy()
roms_v = roms_v.drop_vars(['temp', 'salt', 'u', 'zeta', 'pm', 'pn', 'f'])
roms_v = roms_v.drop_vars(['s_rho', 's_w', 'x_rho', 'y_u', 'y_rho', 'x_u', 'x_psi', 'y_psi'])
roms_v.to_netcdf(os.path.join(OUTDIR, "roms", "v.nc"))

roms_zeta = roms_subset.copy()
roms_zeta = roms_zeta.drop_vars(['temp', 'salt', 'u', 'v', 'pm', 'pn', 'f'])
roms_zeta = roms_zeta.drop_vars(['s_rho', 's_w', 'x_u', 'y_v', 'y_u', 'x_v', 'x_psi', 'y_psi'])
roms_zeta.to_netcdf(os.path.join(OUTDIR, "roms", "zeta.nc"))

roms_pm = roms_subset.copy()
roms_pm = roms_pm.drop_vars(['temp', 'salt', 'u', 'v', 'zeta', 'pn', 'f'])
roms_pm = roms_pm.drop_vars(['s_rho', 's_w', 'x_u', 'y_v', 'y_u', 'x_v', 'x_psi', 'y_psi'])
roms_pm.to_netcdf(os.path.join(OUTDIR, "roms", "pm.nc"))

roms_pn = roms_subset.copy()
roms_pn = roms_pn.drop_vars(['temp', 'salt', 'u', 'v', 'zeta', 'pm', 'f'])
roms_pn = roms_pn.drop_vars(['s_rho', 's_w', 'x_u', 'y_v', 'y_u', 'x_v', 'x_psi', 'y_psi'])	
roms_pn.to_netcdf(os.path.join(OUTDIR, "roms", "pn.nc"))

roms_f = roms_subset.copy()
roms_f = roms_f.drop_vars(['temp', 'salt', 'u', 'v', 'zeta', 'pm', 'pn'])
roms_f = roms_f.drop_vars(['s_rho', 's_w', 'x_u', 'y_v', 'y_u', 'x_v', 'x_psi', 'y_psi'])	
roms_f.to_netcdf(os.path.join(OUTDIR, "roms", "f.nc"))