"""Download IFS reference forecasts from ECMWF open data into notebook-ready NetCDF files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.paths import ifs_root
from common.schedule import build_schedule, format_datetime, generate_init_times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Download IFS open-data forecasts for the notebook factual reference path.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--start', help='First initialization time (YYYYMMDDHH).')
    parser.add_argument('--end', help='Last initialization time (YYYYMMDDHH, inclusive).')
    parser.add_argument(
        '--delta-hours',
        type=int,
        default=12,
        help='Spacing between initialization times. IFS open data is typically run every 12 hours.',
    )
    parser.add_argument(
        '--steps',
        nargs='+',
        type=int,
        default=None,
        help='Explicit forecast lead times to request, in hours.',
    )
    parser.add_argument(
        '--max-lead-hours',
        type=int,
        default=240,
        help='Maximum lead time when --steps is not provided.',
    )
    parser.add_argument(
        '--lead-step-hours',
        type=int,
        default=6,
        help='Lead-time spacing when --steps is not provided.',
    )
    parser.add_argument(
        '--wind-level',
        choices=['850', 'surface'],
        default='850',
        help='Download winds at 850 hPa or 10 m. The current notebook default uses 850 hPa.',
    )
    parser.add_argument(
        '--output-root',
        default=str(ifs_root()),
        help='Root directory where YYYYMMDDHH subdirectories will be created.',
    )
    parser.add_argument(
        '--source',
        choices=['ecmwf', 'aws'],
        default='ecmwf',
        help='ECMWF open-data mirror to query.',
    )
    parser.add_argument('--model', default='ifs', help='Model key for the open-data client.')
    parser.add_argument('--resolution', default='0p25', help='Open-data resolution key.')
    parser.add_argument('--stream', default='oper', help='Forecast stream to request.')
    parser.add_argument('--type', default='fc', help='Product type to request.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing NetCDF files.')
    parser.add_argument('--keep-grib', action='store_true', help='Keep intermediate GRIB files after conversion.')
    return parser.parse_args()


def default_steps(max_lead_hours: int, lead_step_hours: int) -> list[int]:
    if lead_step_hours <= 0:
        raise ValueError('--lead-step-hours must be positive.')
    if max_lead_hours < 0:
        raise ValueError('--max-lead-hours must be non-negative.')
    return list(range(0, max_lead_hours + lead_step_hours, lead_step_hours))


def build_fields(wind_level: str) -> list[dict[str, object]]:
    fields: list[dict[str, object]] = [
        {
            'output_var': 'msl',
            'request_param': 'msl',
            'levtype': 'sfc',
            'levelist': None,
            'source_candidates': ['msl'],
        },
        {
            'output_var': 'q',
            'request_param': 'q',
            'levtype': 'pl',
            'levelist': ['850'],
            'source_candidates': ['q'],
        },
    ]

    if wind_level == '850':
        wind_specs = [
            ('u', 'u', ['u']),
            ('v', 'v', ['v']),
        ]
        levtype = 'pl'
        levelist = ['850']
    else:
        wind_specs = [
            ('u', '10u', ['u10', '10u', 'u']),
            ('v', '10v', ['v10', '10v', 'v']),
        ]
        levtype = 'sfc'
        levelist = None

    for output_var, request_param, source_candidates in wind_specs:
        fields.append(
            {
                'output_var': output_var,
                'request_param': request_param,
                'levtype': levtype,
                'levelist': levelist,
                'source_candidates': source_candidates,
            }
        )

    return fields


def request_grib(client, init_time, steps: list[int], field: dict[str, object], grib_path: Path, args: argparse.Namespace) -> None:
    request = {
        'date': init_time.strftime('%Y-%m-%d'),
        'time': init_time.hour,
        'stream': args.stream,
        'type': args.type,
        'param': [field['request_param']],
        'step': steps,
        'target': str(grib_path),
        'levtype': field['levtype'],
    }
    levelist = field['levelist']
    if levelist is not None:
        request['levelist'] = levelist
    client.retrieve(**request)


def choose_data_var(ds, candidates: list[str]) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise ValueError(f'Could not infer data variable. Available: {list(ds.data_vars)}')


def convert_grib_to_netcdf(grib_path: Path, netcdf_path: Path, field: dict[str, object]) -> None:
    import xarray as xr

    backend_kwargs = {'indexpath': ''}
    with xr.open_dataset(grib_path, engine='cfgrib', backend_kwargs=backend_kwargs) as ds:
        source_var = choose_data_var(ds, field['source_candidates'])
        ds_out = ds[[source_var]].rename({source_var: field['output_var']})
        netcdf_path.parent.mkdir(parents=True, exist_ok=True)
        ds_out.to_netcdf(netcdf_path)


def download_one_init(client, init_time, steps: list[int], fields: list[dict[str, object]], args: argparse.Namespace) -> None:
    init_label = format_datetime(init_time)
    target_dir = Path(args.output_root).expanduser().resolve() / init_label
    target_dir.mkdir(parents=True, exist_ok=True)

    for field in fields:
        output_var = str(field['output_var'])
        grib_path = target_dir / f'{output_var}_ifs_{init_label}.grib'
        netcdf_path = target_dir / f'{output_var}_ifs_{init_label}.nc'

        if netcdf_path.exists() and not args.overwrite:
            print(f'[{init_label}] {netcdf_path.name} exists, skipping.')
            continue

        print(f'[{init_label}] downloading {output_var} -> {grib_path.name}')
        request_grib(client, init_time, steps, field, grib_path, args)
        print(f'[{init_label}] converting {grib_path.name} -> {netcdf_path.name}')
        convert_grib_to_netcdf(grib_path, netcdf_path, field)

        if not args.keep_grib:
            grib_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    schedule = build_schedule(
        start=args.start,
        end=args.end,
        delta_hours=args.delta_hours,
        env_prefix='CYCLONE_IFS',
        default_delta_hours=12,
    )
    init_times = generate_init_times(schedule)
    steps = args.steps or default_steps(args.max_lead_hours, args.lead_step_hours)
    fields = build_fields(args.wind_level)

    try:
        from ecmwf.opendata import Client as OpendataClient
    except ImportError as exc:
        print('The ecmwf-opendata package is required to run this downloader.', file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    client = OpendataClient(source=args.source, model=args.model, resol=args.resolution)

    for init_time in init_times:
        try:
            download_one_init(client, init_time, steps, fields, args)
        except Exception as exc:  # noqa: BLE001
            print(f'[{format_datetime(init_time)}] ERROR: {exc}', file=sys.stderr)
            return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
