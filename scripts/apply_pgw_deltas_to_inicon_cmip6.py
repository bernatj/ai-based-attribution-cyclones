"""Apply pseudo global warming deltas to AI-model initial conditions."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.schedule import build_schedule, generate_init_times
from common.paths import (
    ai_input_grib_root,
    ai_input_netcdf_root,
    aifs_root,
    aurora_root,
    cmip6_multimodel_delta_root,
    cmip6_single_model_delta_root,
)


def load_metpy_runtime():
    import metpy
    from metpy.units import units

    return metpy, units


def adjust_geopotential(z, t, r, sp=False, plev_dim_name='isobaricInhPa'):
    metpy, units = load_metpy_runtime()
    R_d = 287.05
    z = z.load()
    t = t.load()
    r = r.load()
    pressure = z[plev_dim_name].broadcast_like(z)
    lnp = np.log(pressure)
    if not sp:
        w = metpy.calc.mixing_ratio_from_relative_humidity(pressure * units.hPa, t * units.K, r / 100)
    else:
        w = r / (1 - r)
    tv = metpy.calc.virtual_temperature(t * units.K, w).values
    z_new = z.copy()
    for k in range(0, len(z[plev_dim_name]) - 1):
        tv_k12 = (tv[k] + tv[k + 1]) / 2
        z_new[k + 1] = z_new[k] - (lnp[k + 1] - lnp[k]) * R_d * tv_k12
    return z_new


def clean_ncview_metadata(ds):
    drop_vars = [var for var in ['expver'] if var in ds.variables]
    if drop_vars:
        ds = ds.drop_vars(drop_vars)
    for var in ds.data_vars:
        ds[var].attrs.pop('coordinates', None)
    ds.attrs.pop('coordinates', None)
    return ds


def smooth_field(da, lat_dim='latitude', lon_dim='longitude', lat_window=5, lon_window=5):
    if lat_dim in da.dims and lat_window > 1:
        da = da.rolling({lat_dim: lat_window}, center=True, min_periods=1).mean()
    if lon_dim in da.dims and lon_window > 1:
        da = da.rolling({lon_dim: lon_window}, center=True, min_periods=1).mean()
    return da


def get_mode_sign(mode):
    try:
        return {'substract': -1, 'add': 1}[mode]
    except KeyError as exc:
        raise ValueError("Unsupported mode. Use 'substract' or 'add'.") from exc


def standardize_longitudes(ds):
    return ds.assign_coords(longitude=ds['longitude'] % 360).sortby('longitude')


def load_surface_dataset(filepath, grib):
    open_kwargs = {'engine': 'cfgrib'} if grib else {}
    ds = xr.open_dataset(filepath, **open_kwargs)
    ds = ds.squeeze()
    return standardize_longitudes(ds)


def load_pressure_dataset(filepath, grib, plev_dim_name):
    open_kwargs = {'engine': 'cfgrib'} if grib else {}
    ds = xr.open_dataset(filepath, **open_kwargs).squeeze()
    ds = ds.sortby(plev_dim_name, ascending=False)
    ds = standardize_longitudes(ds)
    transpose_order = [dim for dim in [plev_dim_name, 'latitude', 'longitude'] if dim in ds.dims]
    transpose_order += [dim for dim in ds.dims if dim not in transpose_order]
    return ds.transpose(*transpose_order)


def apply_surface_deltas(surface_ds, ds_cmip_deltas, surf_vars, factor, mode_sign):
    for var in surf_vars:
        if var not in ds_cmip_deltas:
            print(f"WARNING: Delta field '{var}' missing; skipping surface update.")
            continue
        if var not in surface_ds:
            print(f"WARNING: Surface variable '{var}' not present in initial conditions; skipping.")
            continue
        delta = ds_cmip_deltas[var] * factor
        if var == 'skt':
            delta_valid = xr.where((delta >= 0) & (delta <= 400), delta, 0)
            valid_mask = surface_ds[var].notnull() & (delta_valid != 0)
            update = mode_sign * delta_valid
            surface_ds[var] = xr.where(valid_mask, surface_ds[var] + update, surface_ds[var])
            continue
        surface_ds[var] = surface_ds[var] + mode_sign * delta
    return surface_ds


def apply_pressure_deltas(pressure_ds, ds_cmip_deltas, pl_vars, factor, mode_sign, plev_dim_name):
    for var in pl_vars:
        if var not in ds_cmip_deltas:
            print(f"WARNING: Delta field '{var}' missing; skipping pressure-level update.")
            continue
        if var not in pressure_ds:
            print(f"WARNING: Pressure-level variable '{var}' not present in initial conditions; skipping.")
            continue
        delta = ds_cmip_deltas[var] * factor
        if 'level' in delta.dims and plev_dim_name != 'level':
            delta = delta.rename({'level': plev_dim_name})
        pressure_ds[var] = pressure_ds[var] + mode_sign * delta
    return pressure_ds


def apply_delta_to_initial_condition_cmip(
    initial_condition_files,
    ds_cmip_deltas,
    surf_vars,
    pl_vars,
    *,
    mode='substract',
    factor=1,
    grib=True,
    plev_dim_name='isobaricInhPa',
):
    if grib:
        initial_condition_S = xr.open_dataset(initial_condition_files['surface'], engine='cfgrib')
    else:
        initial_condition_S = xr.open_dataset(initial_condition_files['surface']).squeeze()
    initial_condition_S = (
        initial_condition_S.assign_coords(longitude=initial_condition_S['longitude'] % 360).sortby('longitude')
    )

    for var in surf_vars:
        if var not in ds_cmip_deltas:
            print(f"WARNING: Delta field '{var}' missing; skipping surface update.")
            continue
        if var not in initial_condition_S:
            print(f"WARNING: Surface variable '{var}' not present in initial conditions; skipping.")
            continue
        delta = ds_cmip_deltas[var] * factor
        if var == 'skt':
            delta_within_range = (delta >= 0) & (delta <= 400)
            valid_mask = initial_condition_S[var].notnull() & delta_within_range
            update = delta if mode == 'add' else -delta
            initial_condition_S[var] = xr.where(valid_mask, initial_condition_S[var] + update, initial_condition_S[var])
            continue
        if mode == 'substract':
            initial_condition_S[var] -= delta
        else:
            initial_condition_S[var] += delta

    open_kwargs = {'engine': 'cfgrib'} if grib else {}
    initial_condition_P = (
        xr.open_dataset(initial_condition_files['pressure'], **open_kwargs)
        .squeeze()
        .sortby(plev_dim_name, ascending=False)
    )
    initial_condition_P = (
        initial_condition_P.assign_coords(longitude=initial_condition_P['longitude'] % 360).sortby('longitude')
    )
    transpose_order = [dim for dim in [plev_dim_name, 'latitude', 'longitude'] if dim in initial_condition_P.dims]
    transpose_order += [dim for dim in initial_condition_P.dims if dim not in transpose_order]
    initial_condition_P = initial_condition_P.transpose(*transpose_order)

    moisture_var = next((var for var in ['q', 'r'] if var in pl_vars and var in initial_condition_P.data_vars), None)
    if moisture_var:
        sp_opt = moisture_var != 'r'
        z_baro_before = adjust_geopotential(
            initial_condition_P['z'],
            initial_condition_P['t'],
            initial_condition_P[moisture_var],
            sp=sp_opt,
            plev_dim_name=plev_dim_name,
        )

    for var in pl_vars:
        if var not in ds_cmip_deltas:
            print(f"WARNING: Delta field '{var}' missing; skipping pressure-level update.")
            continue
        if var not in initial_condition_P:
            print(f"WARNING: Pressure-level variable '{var}' not present in initial conditions; skipping.")
            continue
        delta = ds_cmip_deltas[var].rename({'level': plev_dim_name}) * factor
        if mode == 'substract':
            initial_condition_P[var] -= delta
        else:
            initial_condition_P[var] += delta

    if moisture_var:
        sp_opt = moisture_var != 'r'
        z_baro_after = adjust_geopotential(
            initial_condition_P['z'],
            initial_condition_P['t'],
            initial_condition_P[moisture_var],
            sp=sp_opt,
            plev_dim_name=plev_dim_name,
        )
        initial_condition_P['z'] -= (z_baro_before - z_baro_after)

    if 'valid_time' in initial_condition_S.dims:
        initial_condition_S = initial_condition_S.transpose('valid_time', ...)
    if 'valid_time' in initial_condition_P.dims:
        initial_condition_P = initial_condition_P.transpose('valid_time', ...)

    initial_condition_S = clean_ncview_metadata(initial_condition_S)
    initial_condition_P = clean_ncview_metadata(initial_condition_P)
    return initial_condition_S, initial_condition_P


def fix_cmip6_data(delta_files, levels, ai_model):
    dict_vars = {'t2m': 'tas', 'tcwv': 'prw', 't': 'ta', 'r': 'hur', 'q': 'hus', 'skt': 'tos'}
    if ai_model == 'aifs':
        dict_vars = {'t2m': 'tas', 'tcw': 'prw', 't': 'ta', 'q': 'hus', 'skt': 'tos'}

    ds_vars: Dict[str, xr.DataArray] = {}
    for var, delta_file in delta_files.items():
        var_cmip6 = dict_vars[var]
        da = xr.open_dataset(delta_file, decode_times=False)[var_cmip6]
        if 'time' in da.dims:
            da = da.assign_coords(time=np.arange(da.sizes['time']))
        ds_vars[var] = da

    merged = xr.merge(list(ds_vars.values()))
    reversed_dict = {v: k for k, v in dict_vars.items() if v in merged.data_vars}
    merged = merged.rename(reversed_dict)
    merged['plev'] = merged['plev'] / 100
    merged['plev'].attrs['units'] = 'hPa'

    interpolated_ds = xr.Dataset()
    for var_name in merged.data_vars:
        data_var = merged[var_name]
        if 'plev' in data_var.dims:
            interpolated_ds[var_name] = data_var.interp(plev=levels)
        interpolated_ds[var_name] = data_var
    interpolated_ds['plev'] = levels
    interpolated_ds = interpolated_ds.rename({'plev': 'level'})

    new_lons = np.arange(0, 360, 0.25)
    new_lats = np.arange(90, -90.1, -0.25)
    interpolated_grid_ds = interpolated_ds.interp(
        lon=new_lons, lat=new_lats, method='linear', kwargs={'fill_value': 'extrapolate'}
    )
    interpolated_grid_ds = interpolated_grid_ds.rename({'lat': 'latitude', 'lon': 'longitude'})

    if 'skt' in interpolated_grid_ds.data_vars:
        interpolated_grid_ds['skt'] = smooth_field(interpolated_grid_ds['skt'])
    return interpolated_grid_ds


def interpolate_to_dayofyear(data, day_of_year, method='linear'):
    num_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    dayofyear = [(num_days[i] // 2) + 1 + sum(num_days[0:i]) for i in range(0, 12)]
    data = data.rename({'time': 'dayofyear'})
    data_ext = data.pad(dayofyear=1, mode='wrap')
    dayofyear_ext = [dayofyear[0] - 30] + dayofyear + [dayofyear[-1] + 30]
    data_ext = data_ext.assign_coords(dayofyear=('dayofyear', dayofyear_ext))
    return data_ext.interp(dayofyear=day_of_year, method=method)


def get_model_params(ai_model, path_delta_mm, delta_file_name, modify_skt=False):
    delta_files: Dict[str, str] = {}
    surf_vars: List[str] = []
    pl_vars: List[str] = []

    if ai_model in ['fcnv2']:
        delta_files.update({
            't2m': f"{path_delta_mm}tas/tas_{delta_file_name}.nc",
            'tcwv': f"{path_delta_mm}prw/prw_{delta_file_name}.nc",
            't': f"{path_delta_mm}ta/ta_{delta_file_name}.nc",
            'r': f"{path_delta_mm}hur/hur_{delta_file_name}.nc",
        })
        surf_vars.extend(['t2m', 'tcwv'])
        pl_vars.extend(['t', 'r'])

    if ai_model in ['pangu', 'aurora']:
        delta_files.update({
            't2m': f"{path_delta_mm}tas/tas_{delta_file_name}.nc",
            't': f"{path_delta_mm}ta/ta_{delta_file_name}.nc",
            'q': f"{path_delta_mm}hus/hus_{delta_file_name}.nc",
        })
        surf_vars.append('t2m')
        pl_vars.extend(['t', 'q'])

    if ai_model in ['aifs']:
        delta_files.update({
            't2m': f"{path_delta_mm}tas/tas_{delta_file_name}.nc",
            'tcw': f"{path_delta_mm}prw/prw_{delta_file_name}.nc",
            't': f"{path_delta_mm}ta/ta_{delta_file_name}.nc",
            'q': f"{path_delta_mm}hus/hus_{delta_file_name}.nc",
        })
        surf_vars.extend(['t2m', 'tcw'])
        pl_vars.extend(['t', 'q'])

    if modify_skt:
        delta_files['skt'] = f"{path_delta_mm}tos/tos_{delta_file_name}.nc"
        surf_vars.append('skt')

    return list(set(surf_vars)), list(set(pl_vars)), delta_files



DEFAULT_SINGLE_MODEL_LIST = ['ec-earth3', 'ec-earth3-veg', 'iitm-esm', 'inm-cm5-0',
          'awi-cm-1-1-mr', 'awi-esm-1-1-lr', 'bcc-csm2-mr', 'bcc-esm1',
          'cams-csm1-0', 'cas-esm2-0', 'cmcc-cm2-hr4', 'cmcc-cm2-sr5', 
          'cmcc-esm2', 'canesm5', 'canesm5-1', 'ec-earth3-aerchem', 
          'ec-earth3-cc', 'ec-earth3-veg-lr', 'fio-esm-2-0', 'inm-cm4-8', 
          'kiost-esm', 'mpi-esm-1-2-ham', 'mpi-esm1-2-hr', 'mpi-esm1-2-lr', 
          'nesm3', 'noresm2-lm', 'noresm2-mm', 'taiesm1', 'e3sm-1-1-eca']


def _with_trailing_slash(value: str | None) -> str | None:
    if value is None or value.endswith('/'):
        return value
    return f'{value}/'


def resolve_paths(ai_model, grib_opt, outputdir, path_ic, plev_dim_name):
    fallback_path_ic = None

    if outputdir is None:
        if ai_model == 'aifs':
            outputdir = str(aifs_root() / 'input')
        elif ai_model == 'aurora':
            outputdir = str(aurora_root() / 'input')
        else:
            outputdir = str(ai_input_netcdf_root() if not grib_opt else ai_input_grib_root())

    if path_ic is None:
        if ai_model == 'aifs':
            path_ic = str(aifs_root() / 'input')
            plev_dim_name = plev_dim_name or 'pressure_level'
        elif ai_model == 'aurora':
            path_ic = str(aurora_root() / 'input')
            plev_dim_name = plev_dim_name or 'pressure_level'
        else:
            if grib_opt:
                path_ic = str(ai_input_grib_root())
                fallback_path_ic = str(ai_input_netcdf_root())
            else:
                path_ic = str(ai_input_netcdf_root())
                fallback_path_ic = str(ai_input_grib_root())
    else:
        path_ic = _with_trailing_slash(path_ic)
        if ai_model == 'aifs':
            plev_dim_name = plev_dim_name or 'pressure_level'
        elif ai_model == 'aurora':
            plev_dim_name = plev_dim_name or 'pressure_level'

    path_ic = _with_trailing_slash(path_ic)
    fallback_path_ic = _with_trailing_slash(fallback_path_ic)

    if ai_model in ['aifs', 'aurora']:
        fallback_path_ic = None

    if plev_dim_name is None:
        plev_dim_name = 'isobaricInhPa'

    return outputdir, path_ic, fallback_path_ic, plev_dim_name


def build_initial_condition_files(ai_model, path_ic, yyyymmddhh, ending, alias=None):
    if ai_model == 'aurora':
        return {
            'surface': f"{path_ic}{yyyymmddhh}/{yyyymmddhh}-surface-level.{ending}",
            'pressure': f"{path_ic}{yyyymmddhh}/{yyyymmddhh}-atmospheric.{ending}",
        }
    if ai_model == 'aifs':
        return {
            'surface': f"{path_ic}{yyyymmddhh}/{yyyymmddhh}-surface.{ending}",
            'pressure': f"{path_ic}{yyyymmddhh}/{yyyymmddhh}-pressure-levels.{ending}",
        }
    prefix = alias or ai_model
    return {
        'surface': f"{path_ic}{yyyymmddhh}/{prefix}_sl_{yyyymmddhh}.{ending}",
        'pressure': f"{path_ic}{yyyymmddhh}/{prefix}_pl_{yyyymmddhh}.{ending}",
    }


def get_model_aliases(ai_model: str) -> List[str]:
    aliases = {
        'pangu': ['pangu', 'fcnv2_pangu'],
        'fcnv2': ['fcnv2', 'fcnv2_pangu'],
    }
    return aliases.get(ai_model, [ai_model])


def locate_initial_condition_files(ai_model, yyyymmddhh, primary_path, primary_ending, fallback_path=None, fallback_ending=None):
    search_targets: List[Tuple[str, str, bool]] = []
    if primary_path:
        search_targets.append((primary_path, primary_ending, primary_ending == 'grib'))
    if fallback_path and fallback_ending:
        search_targets.append((fallback_path, fallback_ending, fallback_ending == 'grib'))

    checked_paths: List[str] = []
    for alias in get_model_aliases(ai_model):
        for base_path, ending, is_grib in search_targets:
            candidate_files = build_initial_condition_files(ai_model, base_path, yyyymmddhh, ending, alias=alias)
            missing = [path for path in candidate_files.values() if not os.path.exists(path)]
            checked_paths.extend(candidate_files.values())
            if not missing:
                return candidate_files, is_grib

    search_desc = "\n  ".join(checked_paths) if checked_paths else "  <no paths configured>"
    raise FileNotFoundError(f"Could not find initial-condition files for {yyyymmddhh}. Checked paths:\n  {search_desc}")


def save_modified_files(ai_model, outputdir, yyyymmddhh, exp_label, mod_surface, mod_pressure):
    os.makedirs(f"{outputdir}/{yyyymmddhh}", exist_ok=True)
    if ai_model == 'aurora':
        mod_surface.to_netcdf(f"{outputdir}/{yyyymmddhh}/{yyyymmddhh}-surface-level-{exp_label}.nc")
        mod_pressure.to_netcdf(f"{outputdir}/{yyyymmddhh}/{yyyymmddhh}-atmospheric-{exp_label}.nc")
    elif ai_model == 'aifs':
        mod_surface.to_netcdf(f"{outputdir}/{yyyymmddhh}/{yyyymmddhh}-surface-{exp_label}.nc")
        mod_pressure.to_netcdf(f"{outputdir}/{yyyymmddhh}/{yyyymmddhh}-pressure-levels-{exp_label}.nc")
    else:
        mod_surface.to_netcdf(f"{outputdir}/{yyyymmddhh}/{ai_model}_sl_{exp_label}_{yyyymmddhh}.nc")
        mod_pressure.to_netcdf(f"{outputdir}/{yyyymmddhh}/{ai_model}_pl_{exp_label}_{yyyymmddhh}.nc")

def get_expected_output_files(ai_model, outputdir, yyyymmddhh, exp_label):
    if ai_model == 'aurora':
        return (
            f"{outputdir}/{yyyymmddhh}/{yyyymmddhh}-surface-level-{exp_label}.nc",
            f"{outputdir}/{yyyymmddhh}/{yyyymmddhh}-atmospheric-{exp_label}.nc",
        )
    if ai_model == 'aifs':
        return (
            f"{outputdir}/{yyyymmddhh}/{yyyymmddhh}-surface-{exp_label}.nc",
            f"{outputdir}/{yyyymmddhh}/{yyyymmddhh}-pressure-levels-{exp_label}.nc",
        )
    return (
        f"{outputdir}/{yyyymmddhh}/{ai_model}_sl_{exp_label}_{yyyymmddhh}.nc",
        f"{outputdir}/{yyyymmddhh}/{ai_model}_pl_{exp_label}_{yyyymmddhh}.nc",
    )


def run_pgw_pipeline(
    ai_model,
    schedule,
    path_delta_mm,
    delta_file_name,
    outputdir=None,
    modify_skt=False,
    mode='substract',
    factor=1,
    grib_opt=False,
    path_delta='/home/bernatj/Data/postprocessed-cmip6/interpolated-2.5deg-clim/',
    path_ic=None,
    plev_dim_name=None,
    exp_name='PGW_multimodel_v1',
    multimodel=True,
    single_models=None,
):
    # Avoid duplicated suffixes like "..._skt_skt" when the experiment name already includes "_skt".
    exp_label = exp_name
    if modify_skt and not exp_name.endswith('_skt'):
        exp_label = f"{exp_name}_skt"
    outputdir, path_ic, fallback_path_ic, plev_dim_name = resolve_paths(ai_model, grib_opt, outputdir, path_ic, plev_dim_name)
    levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    init_times = generate_init_times(schedule)
    primary_ending = 'grib' if grib_opt else 'nc'
    fallback_ending = None
    if fallback_path_ic:
        fallback_ending = 'grib' if primary_ending == 'nc' else 'nc'

    if multimodel:
        surf_vars, pl_vars, delta_files = get_model_params(ai_model, path_delta_mm, delta_file_name, modify_skt=modify_skt)
        ds_cmip6_deltas = fix_cmip6_data(delta_files, levels, ai_model)

        for date in init_times:
            yyyymmddhh = date.strftime('%Y%m%d%H')
            out_surface, out_pressure = get_expected_output_files(ai_model, outputdir, yyyymmddhh, exp_label)
            if os.path.exists(out_surface) and os.path.exists(out_pressure):
                print(f"Skipping PGW apply for {yyyymmddhh}: outputs already exist.")
                continue
            initial_condition_files, input_is_grib = locate_initial_condition_files(
                ai_model,
                yyyymmddhh,
                path_ic,
                primary_ending,
                fallback_path=fallback_path_ic,
                fallback_ending=fallback_ending,
            )
            day_of_year = date.timetuple().tm_yday
            ds_cmip6_deltas_doy = interpolate_to_dayofyear(ds_cmip6_deltas, day_of_year)
            mod_surface, mod_pressure = apply_delta_to_initial_condition_cmip(
                initial_condition_files,
                ds_cmip6_deltas_doy,
                surf_vars,
                pl_vars,
                mode=mode,
                factor=factor,
                grib=input_is_grib,
                plev_dim_name=plev_dim_name,
            )
            save_modified_files(ai_model, outputdir, yyyymmddhh, exp_label, mod_surface, mod_pressure)
    else:
        single_models = single_models or DEFAULT_SINGLE_MODEL_LIST
        for model in single_models:
            delta_files = {
                't2m': f"{path_delta}tas/tas_{model}_delta.nc",
                'tcwv': f"{path_delta}prw/prw_{model}_delta.nc",
                't': f"{path_delta}ta/ta_{model}_delta.nc",
                'r': f"{path_delta}hur/hur_{model}_delta.nc",
            }
            if modify_skt:
                delta_files['skt'] = f"{path_delta}tos/tos_{model}_delta.nc"

            try:
                ds_cmip6_deltas = fix_cmip6_data(delta_files, levels, ai_model)
            except Exception:
                print(f'INFO: model files not available for model {model}')
                continue

            for date in init_times:
                yyyymmddhh = date.strftime('%Y%m%d%H')
                out_surface = f"{outputdir}/{yyyymmddhh}/fcnv2_sl_PGW_{model}_{yyyymmddhh}.nc"
                out_pressure = f"{outputdir}/{yyyymmddhh}/fcnv2_pl_PGW_{model}_{yyyymmddhh}.nc"
                if os.path.exists(out_surface) and os.path.exists(out_pressure):
                    print(f"Skipping PGW apply for {yyyymmddhh} ({model}): outputs already exist.")
                    continue
                initial_condition_files = {
                    'surface': f"{path_ic}{yyyymmddhh}/fcnv2_sl_{yyyymmddhh}.grib",
                    'pressure': f"{path_ic}{yyyymmddhh}/fcnv2_pl_{yyyymmddhh}.grib",
                }
                day_of_year = date.timetuple().tm_yday
                ds_cmip6_deltas_doy = interpolate_to_dayofyear(ds_cmip6_deltas, day_of_year)
                mod_surface, mod_pressure = apply_delta_to_initial_condition_cmip(initial_condition_files, ds_cmip6_deltas_doy)
                os.makedirs(f"{outputdir}/{yyyymmddhh}", exist_ok=True)
                mod_surface.to_netcdf(f"{outputdir}/{yyyymmddhh}/fcnv2_sl_PGW_{model}_{yyyymmddhh}.nc")
                mod_pressure.to_netcdf(f"{outputdir}/{yyyymmddhh}/fcnv2_pl_PGW_{model}_{yyyymmddhh}.nc")


def parse_args():
    parser = argparse.ArgumentParser(description='Apply PGW deltas to AI initial conditions.')
    parser.add_argument('--ai-model', required=True, help='Target AI model (e.g., pangu, fcnv2, aurora, aifs).')
    parser.add_argument('--start', help='Start datetime YYYYMMDDHH (overrides env).')
    parser.add_argument('--end', help='End datetime YYYYMMDDHH (overrides env).')
    parser.add_argument('--delta-hours', type=int, help='Hours between init times (overrides env).')
    parser.add_argument('--path-delta-mm', default=str(cmip6_multimodel_delta_root()), help='Directory with multimodel deltas.')
    parser.add_argument('--delta-file-name', default='multimodel_mean_10models_v1', help='Base name for multimodel files.')
    parser.add_argument('--outputdir', default=None, help='Override output directory.')
    parser.add_argument('--modify-skt', action='store_true', help='Apply SST deltas to SKT.')
    parser.add_argument('--mode', choices=['substract', 'add'], default='substract', help='Delta application mode.')
    parser.add_argument('--factor', type=float, default=1.0, help='Delta scaling factor.')
    parser.add_argument('--grib', action='store_true', help='Expect GRIB initial conditions (default NetCDF).')
    parser.add_argument('--path-delta', default=str(cmip6_single_model_delta_root()), help='Directory for single-model deltas.')
    parser.add_argument('--path-ic', default=None, help='Initial condition directory override.')
    parser.add_argument('--plev-dim', default=None, help='Pressure level dimension name override.')
    parser.add_argument('--exp-name', default='PGW_multimodel_v1', help='Experiment label used for outputs.')
    parser.add_argument('--single-model', action='append', help='Specific single-model delta sources (for --no-multimodel).')
    parser.add_argument('--no-multimodel', action='store_true', help='Use individual model deltas instead of multimodel mean.')
    return parser.parse_args()


def main():
    args = parse_args()
    schedule = build_schedule(
        start=args.start,
        end=args.end,
        delta_hours=args.delta_hours,
        default_start=None,
        default_end=None,
    )
    run_pgw_pipeline(
        ai_model=args.ai_model,
        schedule=schedule,
        path_delta_mm=args.path_delta_mm,
        delta_file_name=args.delta_file_name,
        outputdir=args.outputdir,
        modify_skt=args.modify_skt,
        mode=args.mode,
        factor=args.factor,
        grib_opt=args.grib,
        path_delta=args.path_delta,
        path_ic=args.path_ic,
        plev_dim_name=args.plev_dim,
        exp_name=args.exp_name,
        multimodel=not args.no_multimodel,
        single_models=args.single_model,
    )


if __name__ == '__main__':
    main()
