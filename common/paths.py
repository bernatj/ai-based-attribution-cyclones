"""Shared path defaults for the cyclone reproduction repository."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path('/home/bernatj/Data')
DEFAULT_CMIP6_MULTIMODEL_DELTA_ROOT = REPO_ROOT / 'cmip6_deltas'


def _from_env(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    if value:
        return Path(value).expanduser()
    return default


def data_root() -> Path:
    return _from_env('CYCLONE_DATA_ROOT', DEFAULT_DATA_ROOT)


def ai_forecast_root() -> Path:
    return _from_env('CYCLONE_AI_FORECAST_ROOT', data_root() / 'ai-forecasts')


def ai_input_root() -> Path:
    return ai_forecast_root() / 'input'


def ai_input_grib_root() -> Path:
    return ai_input_root() / 'grib'


def ai_input_netcdf_root() -> Path:
    return ai_input_root() / 'netcdf'


def ai_output_root() -> Path:
    return ai_forecast_root() / 'fcst'


def aifs_root() -> Path:
    return _from_env('CYCLONE_AIFS_ROOT', data_root() / 'AIFS_forecasts')


def aurora_root() -> Path:
    return _from_env('CYCLONE_AURORA_ROOT', data_root() / 'Aurora_forecasts')


def ifs_root() -> Path:
    default_root = data_root() / 'IFS_forecasts'
    legacy_root = data_root() / 'ifs-forecats'
    if not default_root.exists() and legacy_root.exists():
        default_root = legacy_root
    return _from_env('CYCLONE_IFS_ROOT', default_root)


def era5_precip_root() -> Path:
    return _from_env('CYCLONE_ERA5_PRECIP_ROOT', data_root() / 'precip_cyclones_era5')


def cmip6_multimodel_delta_root() -> Path:
    default_root = DEFAULT_CMIP6_MULTIMODEL_DELTA_ROOT
    if not default_root.exists():
        default_root = data_root() / 'postprocessed-cmip6' / 'climatology-interpolated-2p5deg-deltas'
    return _from_env(
        'CYCLONE_CMIP6_DELTA_ROOT',
        default_root,
    )


def cmip6_single_model_delta_root() -> Path:
    return _from_env(
        'CYCLONE_CMIP6_SINGLE_MODEL_DELTA_ROOT',
        data_root() / 'postprocessed-cmip6' / 'interpolated-2.5deg-clim',
    )
