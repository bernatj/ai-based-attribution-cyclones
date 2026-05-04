"""Download ERA5 inputs required by the Aurora workflow."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.schedule import build_schedule, generate_init_times
from common.paths import aurora_root

AURORA_LEVELS = ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000']


def download_static_file(download_path: Path, client) -> None:
    static_file = download_path / 'static.nc'
    if static_file.exists():
        return
    static_file.parent.mkdir(parents=True, exist_ok=True)
    print('Downloading Aurora static data ...')
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': ['geopotential', 'land_sea_mask', 'soil_type'],
            'year': '2000',
            'month': '01',
            'day': '01',
            'time': '00:00',
            'format': 'netcdf',
        },
        str(static_file),
    )


def download_for_timestep(date: dt.datetime, download_path: Path, client) -> None:
    yyyymmddhh = date.strftime('%Y%m%d%H')
    date_path = download_path / yyyymmddhh
    date_path.mkdir(parents=True, exist_ok=True)

    surface_file = date_path / f'{yyyymmddhh}-surface-level.nc'
    if not surface_file.exists():
        print(f'Downloading Aurora surface data for {yyyymmddhh} ...')
        client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure'],
                'year': date.strftime('%Y'),
                'month': date.strftime('%m'),
                'day': date.strftime('%d'),
                'time': f'{date.hour:02d}:00',
                'format': 'netcdf',
            },
            str(surface_file),
        )

    atmos_file = date_path / f'{yyyymmddhh}-atmospheric.nc'
    if not atmos_file.exists():
        print(f'Downloading Aurora atmospheric data for {yyyymmddhh} ...')
        client.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': ['temperature', 'u_component_of_wind', 'v_component_of_wind', 'specific_humidity', 'geopotential'],
                'pressure_level': AURORA_LEVELS,
                'year': date.strftime('%Y'),
                'month': date.strftime('%m'),
                'day': date.strftime('%d'),
                'time': f'{date.hour:02d}:00',
                'format': 'netcdf',
            },
            str(atmos_file),
        )


def download_aurora_inputs(schedule, download_path: Path) -> None:
    import cdsapi

    client = cdsapi.Client()
    download_static_file(download_path, client)
    for date in generate_init_times(schedule):
        for ts in [date - dt.timedelta(hours=schedule.delta_hours), date]:
            download_for_timestep(ts, download_path, client)


def parse_args():
    parser = argparse.ArgumentParser(description='Download ERA5 input files required for Aurora runs.')
    parser.add_argument('--output', default=str(aurora_root() / 'input'), help='Input directory.')
    parser.add_argument('--start', help='Start datetime YYYYMMDDHH (overrides env).')
    parser.add_argument('--end', help='End datetime YYYYMMDDHH (overrides env).')
    parser.add_argument('--delta-hours', type=int, help='Hours between initializations (overrides env).')
    return parser.parse_args()


def main():
    args = parse_args()
    schedule = build_schedule(start=args.start, end=args.end, delta_hours=args.delta_hours, default_start=None, default_end=None)
    download_aurora_inputs(schedule, Path(args.output))


if __name__ == '__main__':
    main()
