"""Download ERA5 inputs required by the AIFS workflow."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.schedule import build_schedule, generate_init_times
from common.paths import aifs_root

PARAM_SFC = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    "skin_temperature",
    "surface_pressure",
    "total_column_water",
]
PARAM_SOIL = [
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
]
PARAM_PL = ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind", "vertical_velocity", "specific_humidity"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]


def download_static_file(output_dir: Path, client) -> None:
    static_file = output_dir / 'input' / 'static.nc'
    static_file.parent.mkdir(parents=True, exist_ok=True)
    if static_file.exists():
        return
    print('Downloading static AIFS fields ...')
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': ['land_sea_mask', 'geopotential', 'slope_of_subgridscale_orography', 'standard_deviation_of_orography'],
            'year': '2000',
            'month': '01',
            'day': '01',
            'time': '00:00',
            'format': 'netcdf',
        },
        str(static_file),
    )
    print(f'Static data saved to {static_file}')


def download_init_for_time(date: dt.datetime, output_dir: Path, client) -> None:
    yyyymmddhh = date.strftime('%Y%m%d%H')
    date_dir = output_dir / 'input' / yyyymmddhh
    date_dir.mkdir(parents=True, exist_ok=True)

    surface_file = date_dir / f'{yyyymmddhh}-surface.nc'
    if not surface_file.exists():
        print(f'Downloading surface data for {yyyymmddhh} ...')
        client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': PARAM_SFC,
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'time': f'{date.hour:02d}:00',
                'format': 'netcdf',
            },
            str(surface_file),
        )

    soil_file = date_dir / f'{yyyymmddhh}-soil.nc'
    if not soil_file.exists():
        print(f'Downloading soil data for {yyyymmddhh} ...')
        client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': PARAM_SOIL,
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'time': f'{date.hour:02d}:00',
                'area': [90, -180, -90, 180],
                'grid': [0.25, 0.25],
                'format': 'netcdf',
            },
            str(soil_file),
        )

    pl_file = date_dir / f'{yyyymmddhh}-pressure-levels.nc'
    if not pl_file.exists():
        print(f'Downloading pressure level data for {yyyymmddhh} ...')
        client.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': PARAM_PL,
                'pressure_level': LEVELS,
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'time': f'{date.hour:02d}:00',
                'format': 'netcdf',
            },
            str(pl_file),
        )


def download_aifs_inputs(schedule, output_dir: Path) -> None:
    import cdsapi

    client = cdsapi.Client()
    download_static_file(output_dir, client)
    init_times = generate_init_times(schedule)
    for date in init_times:
        for ts in [date - dt.timedelta(hours=schedule.delta_hours), date]:
            download_init_for_time(ts, output_dir, client)


def parse_args():
    parser = argparse.ArgumentParser(description='Download ERA5 input files required for AIFS runs.')
    parser.add_argument('--output', default=str(aifs_root()), help='Root output directory.')
    parser.add_argument('--start', help='Start datetime YYYYMMDDHH (overrides env).')
    parser.add_argument('--end', help='End datetime YYYYMMDDHH (overrides env).')
    parser.add_argument('--delta-hours', type=int, help='Hours between initializations (overrides env).')
    return parser.parse_args()


def main():
    args = parse_args()
    schedule = build_schedule(start=args.start, end=args.end, delta_hours=args.delta_hours, default_start=None, default_end=None)
    output_dir = Path(args.output)
    download_aifs_inputs(schedule, output_dir)


if __name__ == '__main__':
    main()
