"""Run Earth2MIP-compatible AI models with configurable schedules."""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import List

import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.paths import ai_input_root, ai_output_root

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from common.schedule import build_schedule, generate_init_times

MODEL_ALIASES = {
    'pangu': ['fcnv2_pangu'],
    'fcnv2': ['fcnv2_pangu', 'pangu'],
}

DEFAULT_VARS_TO_SAVE = {
    'pangu': ['u10m', 'v10m', 'msl', 't2m', 't500', 'u850', 'v850', 't850', 'u300', 'v300', 'z500', 'q850'],
    'fcnv2': ['u10m', 'v10m', 'msl', 't2m', 't500', 'u850', 'v850', 't850', 'u300', 'v300', 'z500', 'r850', 'tcwv'],
}


def load_earth2mip_runtime():
    optional_paths = [
        os.getenv('DDF_PATH'),
        os.getenv('EARTH2MIP_PATH'),
        '/home/bernatj/ddf',
        '/home/bernatj/installations/earth2mip',
    ]
    for candidate in optional_paths:
        if candidate and Path(candidate).exists() and candidate not in sys.path:
            sys.path.insert(0, candidate)

    try:
        import torch.serialization as ts
        from earth2mip.inference_ensemble import run_basic_inference
        from earth2mip.networks import get_model
        from ddf._src.data.local.xrda import LocalDataSourceXArray
        from ruamel.yaml.scalarfloat import ScalarFloat
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Earth2MIP dependencies. Activate the forecast environment or set DDF_PATH/EARTH2MIP_PATH."
        ) from exc

    ts.add_safe_globals([ScalarFloat])
    return run_basic_inference, get_model, LocalDataSourceXArray


def generate_file_paths(base_input_dir, file_format, yyyymmddhh, ai_model, experiment, ending) -> List[str]:
    root = Path(base_input_dir) / file_format / yyyymmddhh
    aliases = [ai_model] + MODEL_ALIASES.get(ai_model, [])
    attempted: List[str] = []

    for alias in aliases:
        candidate = [
            root / f"{alias}_sl{experiment}_{yyyymmddhh}.{ending}",
            root / f"{alias}_pl{experiment}_{yyyymmddhh}.{ending}",
        ]
        attempted.extend(str(path) for path in candidate)
        if all(path.exists() for path in candidate):
            return [str(path) for path in candidate]

    raise FileNotFoundError(
        "Could not find input files for "
        f"{ai_model} ({file_format}, init {yyyymmddhh}). Checked:\n  " + "\n  ".join(attempted)
    )


def do_forecast(time_loop, channel_names, file_paths, pressure_name='isobaricInhPa', engine='netcdf4', num_steps=40, t0=None):
    run_basic_inference, _, LocalDataSourceXArray = load_earth2mip_runtime()
    data_source_xr = LocalDataSourceXArray(
        channel_names=channel_names,
        file_paths=file_paths,
        pressure_name=pressure_name,
        name_convention='short_name',
        engine=engine,
    )
    forecast = run_basic_inference(
        time_loop,
        n=num_steps,
        data_source=data_source_xr,
        time=t0,
    )
    return forecast


def run_model_forecasts(
    ai_model,
    model_name,
    schedule,
    num_steps,
    vars_to_save,
    file_format,
    input_base,
    outputdir,
    experiment,
    device='cuda:0',
):
    _, get_model, _ = load_earth2mip_runtime()
    model_uri = f"e2mip://{model_name}"
    print(f'Loading AI-model: {model_uri}')
    time_loop = get_model(model=model_uri, device=device)
    channel_names = time_loop.in_channel_names
    print('Model loaded successfully')

    init_times = generate_init_times(schedule)
    ending = 'grib' if file_format == 'grib' else 'nc'
    engine = 'cfgrib' if file_format == 'grib' else 'netcdf4'

    for t0 in init_times:
        yyyymmddhh = t0.strftime('%Y%m%d%H')
        print(f'Running forecast for {yyyymmddhh} ...')
        file_paths = generate_file_paths(input_base, file_format, yyyymmddhh, ai_model, experiment, ending)
        forecast = do_forecast(time_loop, channel_names, file_paths, pressure_name='isobaricInhPa', engine=engine, num_steps=num_steps, t0=t0)

        print('Saving forecast fields ...')
        os.makedirs(f"{outputdir}/{yyyymmddhh}", exist_ok=True)
        for var in vars_to_save:
            forecast.sel(channel=var).squeeze().drop_vars('channel').to_dataset(name=var).to_netcdf(
                f"{outputdir}/{yyyymmddhh}/{var}_{ai_model}{experiment}_{yyyymmddhh}.nc",
                mode='w',
            )
        print(f'Finished forecast for init {yyyymmddhh}')


def parse_args():
    parser = argparse.ArgumentParser(description='Generic runner for Earth2MIP inference loops.')
    parser.add_argument('--ai-model', default='pangu', help='Short name used in filenames (default: pangu).')
    parser.add_argument('--model-name', default='pangu', help='Earth2MIP registry model name (default: pangu).')
    parser.add_argument('--start', help='Start datetime YYYYMMDDHH (overrides env).')
    parser.add_argument('--end', help='End datetime YYYYMMDDHH (overrides env).')
    parser.add_argument('--delta-hours', type=int, help='Hours between initializations (overrides env).')
    parser.add_argument('--num-steps', type=int, default=40, help='Number of 6-hour steps to run (default: 40).')
    parser.add_argument('--var', action='append', help='Variables to save (can be repeated).')
    parser.add_argument('--file-format', choices=['netcdf', 'grib'], default='netcdf', help='Input file format (default: netcdf).')
    parser.add_argument('--input-base', default=str(ai_input_root()), help='Base directory for initial conditions.')
    parser.add_argument('--output', default=str(ai_output_root()), help='Forecast output directory.')
    parser.add_argument('--experiment', default='', help='Experiment suffix used in filenames (include leading underscore if desired).')
    parser.add_argument('--device', default='cuda:0', help='Torch device string (default: cuda:0).')
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
    default_vars = DEFAULT_VARS_TO_SAVE.get(args.ai_model, DEFAULT_VARS_TO_SAVE.get('pangu', []))
    vars_to_save = args.var or default_vars
    run_model_forecasts(
        ai_model=args.ai_model,
        model_name=args.model_name,
        schedule=schedule,
        num_steps=args.num_steps,
        vars_to_save=vars_to_save,
        file_format=args.file_format,
        input_base=args.input_base,
        outputdir=args.output,
        experiment=args.experiment,
        device=args.device,
    )


if __name__ == '__main__':
    main()
