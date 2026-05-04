"""Download ERA5 initialization data compatible with several AI models."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import glob

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.schedule import Schedule, build_schedule, generate_init_times
from common.paths import ai_input_grib_root


MODEL_CONFIGS = {
    'fcnv2': {
        'plevels': ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000'],
        'variables_pl': ['129', '130', '131', '132', '157'],
        'variables_sf': ['134', '137', '151', '165', '166', '167', '228246', '228247'],
    },
    'pangu': {
        'plevels': ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000'],
        'variables_pl': ['129', '130', '131', '132', '133'],
        'variables_sf': ['151', '165', '166', '167'],
    },
    'graphcast': {
        'plevels': ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70', '100', '125', '150', '175', '200',
                    '225', '250', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '775',
                    '800', '825', '850', '875', '900', '925', '950', '975', '1000'],
        'variables_pl': ['129', '130', '131', '132', '133', '135'],
        'variables_sf': ['151', '165', '166', '167', '228', '260267'],
    },
}

GRIB_CODE_TO_SHORTNAME = {
    '129': 'z',
    '130': 't',
    '131': 'u',
    '132': 'v',
    '133': 'q',
    '134': 'sp',
    '137': 'tcwv',
    '141': 'sd',
    '151': 'msl',
    '165': 'u10',
    '166': 'v10',
    '167': 't2m',
    '168': 'd2m',
    '228': 'tp',
    '228246': 'tp',
    '228247': 'tcw',
    '260267': 'tisr',
}


def _normalize_model_input(model_input: Iterable[str] | str) -> List[str]:
    if isinstance(model_input, str):
        return [model_input]
    try:
        return list(model_input)
    except TypeError as exc:
        raise ValueError('model_input must be a string or an iterable of strings') from exc


def _intersect_preserve_order(sequences: Sequence[Sequence[str]]) -> List[str]:
    common = set(sequences[0])
    for seq in sequences[1:]:
        common &= set(seq)
    return [value for value in sequences[0] if value in common]


def _union_preserve_order(sequences: Sequence[Sequence[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for seq in sequences:
        for value in seq:
            if value not in seen:
                seen.add(value)
                result.append(value)
    return result


def get_model_configuration(model_input: Iterable[str] | str) -> Tuple[str, List[str], List[str], List[str]]:
    model_list = _normalize_model_input(model_input)
    if not model_list:
        raise ValueError('At least one model must be provided')

    missing = [name for name in model_list if name not in MODEL_CONFIGS]
    if missing:
        raise ValueError(f"Unknown model(s): {', '.join(missing)}")

    if len(model_list) == 1:
        cfg = MODEL_CONFIGS[model_list[0]]
        return model_list[0], cfg['plevels'], cfg['variables_pl'], cfg['variables_sf']

    configs = [MODEL_CONFIGS[name] for name in model_list]
    plevels = _intersect_preserve_order([cfg['plevels'] for cfg in configs])
    variables_pl = _union_preserve_order([cfg['variables_pl'] for cfg in configs])
    variables_sf = _union_preserve_order([cfg['variables_sf'] for cfg in configs])

    if not plevels:
        raise ValueError('Selected models do not share a common set of pressure levels.')

    return '_'.join(sorted(model_list)), plevels, variables_pl, variables_sf


def _list_shortnames(filepath: str) -> List[str]:
    import cfgrib

    datasets = cfgrib.open_datasets(filepath)
    names: List[str] = []
    for ds in datasets:
        for var in ds.data_vars.values():
            short_name = var.attrs.get('GRIB_shortName', var.name)
            if short_name not in names:
                names.append(short_name)
    return names


def _expected_shortnames(codes: List[str]) -> List[str]:
    return [GRIB_CODE_TO_SHORTNAME.get(code, code) for code in codes]


def _missing_variables(filepath: str, codes: List[str]) -> List[str]:
    if not os.path.exists(filepath):
        return _expected_shortnames(codes)
    try:
        available = set(_list_shortnames(filepath))
    except Exception:
        return _expected_shortnames(codes)
    missing = [name for name in _expected_shortnames(codes) if name not in available]
    return missing


def _cleanup_idx_files(outfile: str) -> None:
    pattern = f"{outfile}*.idx"
    for idx in glob.glob(pattern):
        try:
            os.remove(idx)
        except FileNotFoundError:
            pass


def download_init(
    yyyymmddhh: str,
    modelname: str,
    leveltype: str,
    variables: List[str],
    dataset: str,
    plevels: List[str],
    savedir: str,
):
    import cdsapi

    c = cdsapi.Client()
    filename = f'{modelname}_{leveltype}_{yyyymmddhh}.grib'
    outfile = os.path.join(savedir, filename)
    if os.path.exists(outfile):
        missing_vars = _missing_variables(outfile, variables)
        if missing_vars:
            print(f'{filename} exists but is missing variables {missing_vars}. Re-downloading.')
            os.remove(outfile)
            _cleanup_idx_files(outfile)
        else:
            print(f'{filename} already exists. Skipping download.')
            return
    print(f'Downloading init {yyyymmddhh} for {modelname} {leveltype}')

    request_params = {
        'product_type': 'reanalysis',
        'data_format': 'grib',
        'download_format': 'unarchived',
        'variable': variables,
        'year': yyyymmddhh[0:4],
        'month': yyyymmddhh[4:6],
        'day': yyyymmddhh[6:8],
        'time': f"{yyyymmddhh[8:10]}:00",
    }
    if leveltype == 'pl':
        request_params['pressure_level'] = plevels

    try:
        c.retrieve(dataset, request_params, outfile)
    except Exception as exc:
        if os.path.exists(outfile):
            os.remove(outfile)
        _cleanup_idx_files(outfile)
        raise RuntimeError(f"Failed to download {filename}: {exc}") from exc
    print(f'Download complete for {filename}.')


def run_downloads(
    model_input: Iterable[str] | str,
    schedule: Schedule,
    base_savedir: str,
    max_workers: int = 5,
):
    modelname, plevels, variables_pl, variables_sf = get_model_configuration(model_input)
    init_times = generate_init_times(schedule)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for init_time in init_times:
            yyyymmddhh = init_time.strftime('%Y%m%d%H')
            savedir = os.path.join(base_savedir, yyyymmddhh)
            os.makedirs(savedir, exist_ok=True)
            print(f"[download] Queued init {yyyymmddhh} for {modelname}", flush=True)

            futures[
                executor.submit(
                    download_init,
                    yyyymmddhh,
                    modelname,
                    'sl',
                    variables_sf,
                    'reanalysis-era5-single-levels',
                    plevels,
                    savedir,
                )
            ] = (yyyymmddhh, 'surface')
            futures[
                executor.submit(
                    download_init,
                    yyyymmddhh,
                    modelname,
                    'pl',
                    variables_pl,
                    'reanalysis-era5-pressure-levels',
                    plevels,
                    savedir,
                )
            ] = (yyyymmddhh, 'pressure')

        for future in concurrent.futures.as_completed(futures):
            yyyymmddhh, level = futures[future]
            try:
                future.result()
                print(f"[download] Finished {level} init {yyyymmddhh}", flush=True)
            except Exception as exc:
                print(f"[download] FAILED {level} init {yyyymmddhh}: {exc}", flush=True)
                raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download ERA5 initial conditions for AI models.')
    parser.add_argument('--models', default='pangu', help='Comma-separated list of AI model names to intersect.')
    parser.add_argument('--start', help='Start datetime YYYYMMDDHH (overrides env).')
    parser.add_argument('--end', help='End datetime YYYYMMDDHH (overrides env).')
    parser.add_argument('--delta-hours', type=int, help='Delta between inits (overrides env).')
    parser.add_argument('--outdir', default=str(ai_input_grib_root()), help='Base output directory.')
    parser.add_argument('--workers', type=int, default=5, help='Parallel download workers.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[download] AI downloader starting", flush=True)
    model_names = [name.strip() for name in args.models.split(',') if name.strip()]
    schedule = build_schedule(
        start=args.start,
        end=args.end,
        delta_hours=args.delta_hours,
        default_start=None,
        default_end=None,
        env_prefix='AI_DOWNLOAD',
    )
    run_downloads(model_names, schedule, base_savedir=args.outdir, max_workers=args.workers)
    print("[download] All requested downloads completed", flush=True)


if __name__ == '__main__':
    main()
