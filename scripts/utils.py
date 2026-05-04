import datetime

import numpy as np
import pandas as pd
import xarray as xr


def flip_lon_360_2_180(var_360, lon):
    """
    This function shifts the longitude dimension from [0,360] to [-180,180].
    """
    var_180 = var_360.assign_coords(lon=(lon + 180) % 360 - 180)
    var_180 = var_180.sortby(var_180.lon)

    return var_180


def subset_region(da, lat_slice, lon_slice):
    """
    Subset a lat/lon box without remapping the full global longitude grid first.

    For fields stored on a 0-360 longitude grid and a requested box that crosses
    Greenwich (for example -25 to 10), this selects the two native longitude
    segments first and only then converts the smaller subset to [-180, 180].
    """
    lat_name = "lat"
    lon_name = "lon"

    subset = da.sel({lat_name: lat_slice})
    lon = subset[lon_name]
    lon_min = float(lon.min())
    lon_max = float(lon.max())

    if lon_min >= -180 and lon_max <= 180:
        return subset.sel({lon_name: lon_slice})

    lon_w = float(lon_slice.start)
    lon_e = float(lon_slice.stop)

    if lon_w <= lon_e and lon_w < 0:
        left = subset.sel({lon_name: slice(360 + lon_w, lon_max)})
        right = subset.sel({lon_name: slice(0, lon_e)})
        region = xr.concat([left, right], dim=lon_name)
    else:
        native_w = lon_w if lon_w >= 0 else 360 + lon_w
        native_e = lon_e if lon_e >= 0 else 360 + lon_e
        region = subset.sel({lon_name: slice(native_w, native_e)})

    region = region.assign_coords({lon_name: ((region[lon_name] + 180) % 360) - 180})
    return region.sortby(lon_name)


def from_init_time_to_leadtime(var_init_time, init_time_min, lead_time_range, time_range):
    """
    This function creates a xarray DataArray for a given variable and fills it with values based on the provided time range and lead time range.

    Parameters:
    var_init_time (xarray.DataArray): The initial time data for the variable.
    init_time_min (datetime): The minimum initial time.
    lead_time_range (numpy.ndarray): The range of lead times.
    time_range (pandas.DatetimeIndex): The range of times.

    Returns:
    xarray.DataArray: The created and filled DataArray for the variable.
    """

    time_values = pd.DatetimeIndex(pd.to_datetime(time_range))

    if len(time_values) == 0:
        dims = ["lead_time", "time"] + list(var_init_time.dims)[2:]
        coords = {"lead_time": lead_time_range, "time": time_values}
        coords.update({dim: var_init_time.coords[dim] for dim in var_init_time.dims[2:]})
        return xr.DataArray(dims=dims, coords=coords)

    try:
        global_times = pd.Index(pd.to_datetime(var_init_time["time"].values))
        init_times = pd.Index(pd.to_datetime(var_init_time["init_time"].values))
        time_positions = global_times.get_indexer(time_values)
        if np.any(time_positions < 0):
            raise ValueError("Requested times are not present in the source time axis.")

        if len(time_values) > 1:
            step = time_values[1] - time_values[0]
        elif len(lead_time_range) > 1:
            step = pd.to_timedelta(int(lead_time_range[1] - lead_time_range[0]), unit="h")
        else:
            step = pd.to_timedelta(6, unit="h")

        step_hours = step / pd.Timedelta(hours=1)
        if step_hours <= 0:
            raise ValueError("Invalid time step for lead-time conversion.")

        time_indexer = xr.DataArray(time_positions, dims="time", coords={"time": time_values})
        n_init = var_init_time.sizes["init_time"]
        lead_arrays = []

        for lt in lead_time_range:
            lead_steps = int(round(float(lt) / float(step_hours)))
            if not np.isclose(lead_steps * float(step_hours), float(lt)):
                raise ValueError("Lead times are not aligned with the time grid.")

            required_init_times = time_values - pd.to_timedelta(int(lt), unit="h")
            init_positions = init_times.get_indexer(required_init_times)
            valid = (init_positions >= 0) & (init_positions < n_init)
            valid &= required_init_times >= pd.Timestamp(init_time_min)

            safe_init_positions = np.clip(init_positions, 0, max(n_init - 1, 0))
            init_indexer = xr.DataArray(safe_init_positions, dims="time", coords={"time": time_values})
            valid_mask = xr.DataArray(valid, dims="time", coords={"time": time_values})

            lead_da = var_init_time.isel(init_time=init_indexer, time=time_indexer).where(valid_mask)
            if "init_time" in lead_da.coords and "init_time" not in lead_da.dims:
                lead_da = lead_da.drop_vars("init_time")
            lead_arrays.append(lead_da.expand_dims(lead_time=[lt]))

        return xr.concat(lead_arrays, dim="lead_time")
    except Exception:
        dims = ["lead_time", "time"] + list(var_init_time.dims)[2:]
        coords = {"lead_time": lead_time_range, "time": time_values}
        coords.update({dim: var_init_time.coords[dim] for dim in var_init_time.dims[2:]})
        var_leadtime = xr.DataArray(dims=dims, coords=coords)

        for t in time_values:
            for lt in lead_time_range:
                it = t.to_pydatetime() - datetime.timedelta(hours=int(lt))

                if it < init_time_min:
                    var_leadtime.loc[{"lead_time": lt, "time": t}] = np.nan
                else:
                    try:
                        var_leadtime.loc[{"lead_time": lt, "time": t}] = var_init_time.sel(init_time=it, time=t)
                    except Exception:
                        var_leadtime.loc[{"lead_time": lt, "time": t}] = np.nan

        return var_leadtime


def add_country_mask(ds: xr.Dataset, country: str = "Spain") -> xr.Dataset:
    import regionmask

    # get countries mask
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110

    # create mask variable
    mask = countries.mask_3D(ds)

    # select Spain mask
    var_name = country.lower()
    ds[f"{var_name}_mask"] = mask.isel(region=(mask.names == "Spain")).squeeze().astype(np.int16)

    return ds


def add_land_mask(ds: xr.Dataset) -> xr.Dataset:
    import regionmask

    # get land-sea-mask mask
    land_110 = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    # create land mask variable
    ds["land_mask"] = land_110.mask_3D(ds).squeeze().astype(np.int16)

    return ds


def load_data(var, init_times, root, level=None, extension='', model='fcnv2'):
    """
    Load and concatenate data from multiple NetCDF files into a single xarray Dataset.

    Parameters:
        var (str): Variable name to load from NetCDF files.
        init_times (list): List of datetime initialization times.
        root (str): Root directory containing the files.
        level (int/float, optional): Pressure level to select (for models with level dimension).
        extension (str, optional): Optional suffix for file names.
        model (str): Model identifier ('fcnv2', 'neuralgcm', 'aurora', etc.).

    Returns:
        xarray.Dataset: Concatenated dataset with 'init_time' dimension.

    Example:
        >>> init_times = [datetime(2020,1,1), datetime(2020,1,2)]
        >>> data = load_data('z500', init_times, '/data', model='aurora', level=500)
    """
    var_inits = []

    for t0 in init_times:
        yyyymmddhh = t0.strftime('%Y%m%d%H')
        file = f'{root}/{yyyymmddhh}/{var}_{model}_{extension}{yyyymmddhh}.nc'
        print(f"Loading: {file}")

        ds = xr.open_dataset(file, chunks={})[var]

        # Special processing only for neuralgcm
        if model == 'neuralgcm':
            # Process time coordinates
            try:
                new_time = pd.to_timedelta(ds.time, unit='h') + t0
                ds = ds.assign_coords(time=new_time)
            except Exception:
                rounded_time = pd.to_datetime(ds.coords['time'].values).round('h')
                ds = ds.assign_coords(time=rounded_time)

            # Flip latitude and rename coordinates
            ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
            ds = ds.isel(lat=slice(None, None, -1))

        elif model == 'ifs':
            # Rename coordinates to match common convention
            rename_map = {}
            if 'latitude' in ds.dims:
                rename_map['latitude'] = 'lat'
            if 'longitude' in ds.dims:
                rename_map['longitude'] = 'lon'
            if rename_map:
                ds = ds.rename(rename_map)

            if 'isobaricInhPa' in ds.dims or 'isobaricInhPa' in ds.coords:
                ds = ds.rename({'isobaricInhPa': 'level'})

            # Replace step dimension with actual valid time
            if 'time' in ds.coords:
                ds = ds.drop_vars('time')
            lead_name = None
            if 'valid_time' in ds.coords:
                ds = ds.swap_dims({'step': 'valid_time'}).rename({'valid_time': 'time'})
            elif 'step' in ds.dims:
                step_coord = ds['step']
                if np.issubdtype(step_coord.dtype, np.datetime64):
                    ds = ds.rename({'step': 'time'})
                else:
                    # fallback: keep step values but expose them as time for consistency
                    ds = ds.rename({'step': 'time'})
            elif 'lead_time' in ds.dims:
                lead_name = 'lead_time'
            elif 'step' in ds.dims:
                lead_name = 'step'
            if lead_name is not None and 'time' not in ds.dims:
                lead = ds.coords[lead_name]
                lead_hours = xr.DataArray(lead, dims=lead.dims)
                # convert to timedelta64[ns] if not already
                if not np.issubdtype(lead_hours.dtype, np.timedelta64):
                    lead_hours = lead_hours.astype('timedelta64[h]')
                if 'init_time' in ds.coords:
                    abs_time = ds.coords['init_time'] + lead_hours
                else:
                    abs_time = lead_hours
                ds = ds.rename({lead_name: 'time'}).assign_coords(time=abs_time)

            # Keep lead-time coordinate for reference
            if 'step' in ds.coords:
                ds = ds.rename({'step': 'lead_time'})

            # Ensure coordinates are strictly increasing when time coord exists
            if 'time' in ds.coords:
                ds = ds.sortby('time')

        # Optional level selection for all models
        if level is not None and 'level' in ds.dims:
            ds = ds.sel(level=level)

        var_inits.append(ds)

    merged_dataset = xr.concat(var_inits, dim='init_time')
    return merged_dataset.assign_coords(init_time=init_times)


def area_average(var):
    """
    Compute the area-weighted global average of a variable, ignoring missing values.

    This function calculates the global average of a variable using latitude-dependent weights
    while excluding missing values (NaNs) from the calculation.

    Parameters:
        var (xarray.DataArray): Input data with 'lat' and 'lon' dimensions, may contain NaNs.

    Returns:
        xarray.DataArray: Area-weighted global average, averaged over longitude, excluding NaNs.
    """
    # Extract latitude values from the DataArray
    lat = var.lat

    # Compute area weights based on latitude
    weights = np.cos(lat * np.pi / 180)

    # Ensure weights and var are aligned
    weights_da = xr.DataArray(weights, coords=[var.lat], dims=['lat']).broadcast_like(var)
    weights_da = xr.where(np.isnan(var), np.nan, weights_da)

    # Calculate the weighted variable and the total weight
    weighted_var = var * weights_da
    weighted_sum = weighted_var.sum('lat', skipna=True)

    total_weight = weights_da.sum('lat', skipna=True)

    # Calculate the global mean, ignoring NaNs
    area_average = (weighted_sum / total_weight).mean('lon', skipna=True)

    return area_average
