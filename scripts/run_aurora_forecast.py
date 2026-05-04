import argparse
import datetime
import os
from pathlib import Path
import sys

import numpy as np
import xarray as xr

try:
    import cdsapi
    import torch
    from aurora import Aurora, Batch, Metadata, rollout
except ModuleNotFoundError:
    cdsapi = None
    torch = None
    Aurora = None
    Batch = None
    Metadata = None
    rollout = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.schedule import build_schedule, generate_init_times
from common.paths import aurora_root

# ---------------------------
# Configuration
# ---------------------------
DEFAULT_INPUT_DATA_PATH = aurora_root() / 'input'
DEFAULT_OUTPUT_DATA_PATH = aurora_root() / 'fcst'
DEFAULT_FIRST_INIT_TIME = datetime.datetime(2024, 10, 22, 0)
DEFAULT_END_INIT_TIME = datetime.datetime(2024, 11, 1, 18)
DEFAULT_DELTA_HOURS = 6
AURORA_MODEL_NAME = "aurora-0.25-pretrained.ckpt"
 #other model versions are: "aurora-0.25-finetuned.ckpt", "aurora-0.1-finetuned.ckpt" 

DEFAULT_PGW_ENABLED = True  # Set to True to run PGW experiments
DEFAULT_EXP_NAME='PGW_multimodel_v1'  #use this to run PGW experiments
TIME_STEPS = 10 * 4  # 10 days


def require_runtime_dependencies() -> None:
    missing = []
    if cdsapi is None:
        missing.append('cdsapi')
    if torch is None:
        missing.append('torch')
    if Aurora is None or Batch is None or Metadata is None or rollout is None:
        missing.append('aurora')
    if missing:
        joined = ', '.join(missing)
        raise SystemExit(
            f"Missing Aurora runtime dependencies: {joined}. Activate the Aurora environment before running this script."
        )

def download_era5_data(date: datetime, download_path: Path, c):
    # Calculate previous timestep (exactly 6 hours before)
    prev_date = date - datetime.timedelta(hours=6)
    
    # Create directories
    download_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Static variables (one-time download)
    static_file = download_path / "static.nc"
    if not static_file.exists():
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["geopotential", "land_sea_mask", "soil_type"],
                "year": "2000",
                "month": "01",
                "day": "01",
                "time": "00:00",
                "format": "netcdf",
            },
            str(static_file),
        )

    # 2. Download each timestep separately
    for current_date in [prev_date, date]:
        yyyymmddhh = current_date.strftime('%Y%m%d%H')
        date_path = download_path / yyyymmddhh
        date_path.mkdir(exist_ok=True)
        
        # Surface data for this timestep
        surface_file = date_path / f"{yyyymmddhh}-surface-level.nc"
        if not surface_file.exists():
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "2m_temperature",
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "mean_sea_level_pressure"
                    ],
                    "year": current_date.strftime("%Y"),
                    "month": current_date.strftime("%m"),
                    "day": current_date.strftime("%d"),
                    "time": f"{current_date.hour:02d}:00",
                    "format": "netcdf",
                },
                str(surface_file),
            )
            print(f"Downloaded surface data for {yyyymmddhh}")

        # Atmospheric data for this timestep
        atmospheric_file = date_path / f"{yyyymmddhh}-atmospheric.nc"
        if not atmospheric_file.exists():
            c.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "temperature",
                        "u_component_of_wind",
                        "v_component_of_wind",
                        "specific_humidity",
                        "geopotential"
                    ],
                    "pressure_level": ["50", "100", "150", "200", "250", "300", "400", 
                                     "500", "600", "700", "850", "925", "1000"],
                    "year": current_date.strftime("%Y"),
                    "month": current_date.strftime("%m"),
                    "day": current_date.strftime("%d"),
                    "time": f"{current_date.hour:02d}:00",
                    "format": "netcdf",
                },
                str(atmospheric_file),
            )
            print(f"Downloaded atmospheric data for {yyyymmddhh}")

    print(f"Completed downloads for {prev_date.strftime('%Y%m%d%H')} and {date.strftime('%Y%m%d%H')}")

def prepare_batch(download_path: Path, timestamp: datetime, pgw_enabled: bool, experiment_label: str) -> Batch:
    require_runtime_dependencies()
    # Get current and previous timesteps (6h apart)
    prev_timestamp = timestamp - datetime.timedelta(hours=6)
    current_yyyymmddhh = timestamp.strftime('%Y%m%d%H')
    prev_yyyymmddhh = prev_timestamp.strftime('%Y%m%d%H')
    
    # Load static data (unchanged)
    static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
    
    experiment_suffix = ""
    if pgw_enabled:
        experiment_suffix = f"-{experiment_label}"
    # Load surface data from both timesteps
    surface_prev = xr.open_dataset(download_path / prev_yyyymmddhh / f"{prev_yyyymmddhh}-surface-level{experiment_suffix}.nc", engine="netcdf4")
    surface_curr = xr.open_dataset(download_path / current_yyyymmddhh / f"{current_yyyymmddhh}-surface-level{experiment_suffix}.nc", engine="netcdf4")
    surf_vars_ds = xr.concat([surface_prev, surface_curr], dim="valid_time")

    # Load atmospheric data from both timesteps
    atmos_prev = xr.open_dataset(download_path / prev_yyyymmddhh / f"{prev_yyyymmddhh}-atmospheric{experiment_suffix}.nc", engine="netcdf4")
    atmos_curr = xr.open_dataset(download_path / current_yyyymmddhh / f"{current_yyyymmddhh}-atmospheric{experiment_suffix}.nc", engine="netcdf4")
    atmos_vars_ds = xr.concat([atmos_prev, atmos_curr], dim="valid_time")
    
    print(atmos_vars_ds)

    # Prepare batch
    batch = Batch(
        surf_vars={
            "2t": torch.from_numpy(surf_vars_ds["t2m"].values[[0, 1]][None]),
            "10u": torch.from_numpy(surf_vars_ds["u10"].values[[0, 1]][None]),
            "10v": torch.from_numpy(surf_vars_ds["v10"].values[[0, 1]][None]),
            "msl": torch.from_numpy(surf_vars_ds["msl"].values[[0, 1]][None]),
        },
        static_vars={
            "z": torch.from_numpy(static_vars_ds["z"].values[0]),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_vars_ds["t"].values[[0, 1]][None]),
            "u": torch.from_numpy(atmos_vars_ds["u"].values[[0, 1]][None]),
            "v": torch.from_numpy(atmos_vars_ds["v"].values[[0, 1]][None]),
            "q": torch.from_numpy(atmos_vars_ds["q"].values[[0, 1]][None]),
            "z": torch.from_numpy(atmos_vars_ds["z"].values[[0, 1]][None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_vars_ds.latitude.values),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[1],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
        ),
    )
    
    return batch


def save_predictions(
    preds,
    output_path: Path,
    timestamp: datetime,
    pgw_enabled: bool,
    experiment_label: str,
    overwrite: bool = False,
):
    yyyymmddhh = timestamp.strftime('%Y%m%d%H')
    output_path = output_path / yyyymmddhh
    output_path.mkdir(parents=True, exist_ok=True)

    lat = preds[0].metadata.lat.numpy()
    lon = preds[0].metadata.lon.numpy()
    levels = np.array(preds[0].metadata.atmos_levels)

    def save_var(var_type, var_dict, dims):
        for var, _ in var_dict.items():
            file_path = output_path / f"{var}_aurora_{yyyymmddhh}.nc"
            if pgw_enabled:
                file_path = output_path / f"{var}_aurora_{experiment_label}_{yyyymmddhh}.nc"
            if file_path.exists() and not overwrite:
                continue
            values = [pred.__getattribute__(var_type)[var][0, 0].numpy() for pred in preds]
            times = [pred.metadata.time[0] for pred in preds]
            data = np.stack(values)

            coords = {"time": times, "lat": lat, "lon": lon}
            if "level" in dims:
                coords["level"] = levels

            xr.DataArray(data, dims=dims, coords=coords, name=var).to_netcdf(file_path)

    save_var("surf_vars", preds[0].surf_vars, ("time", "lat", "lon"))
    save_var("atmos_vars", preds[0].atmos_vars, ("time", "level", "lat", "lon"))

# ---------------------------
# Main Forecast Loop
# ---------------------------
def run_forecast(
    schedule,
    input_path: Path,
    output_path: Path,
    experiment_label: str,
    pgw_enabled: bool,
    skip_download: bool,
):
    require_runtime_dependencies()
    print("Initializing model...")
    device = os.getenv('CYCLONE_AURORA_DEVICE', 'cuda:0')
    model = Aurora(use_lora=False)
    model.load_checkpoint("microsoft/aurora", AURORA_MODEL_NAME)
    model = model.eval().to(device)

    init_times = generate_init_times(schedule)

    for time in init_times:
        print(f"\nProcessing forecast for: {time}")
        if not skip_download:
            c = cdsapi.Client()
            download_era5_data(time, input_path, c)

        batch = prepare_batch(input_path, time, pgw_enabled, experiment_label)

        print("Running Aurora forecast...")
        with torch.inference_mode():
            preds = [pred.to("cpu") for pred in rollout(model, batch, steps=TIME_STEPS)]

        print("Saving outputs...")
        save_predictions(preds, output_path, time, pgw_enabled, experiment_label, overwrite=True)

    model.to("cpu")
    print("Forecasting complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Aurora forecasts.')
    parser.add_argument('--start', help='Start datetime YYYYMMDDHH (overrides default).')
    parser.add_argument('--end', help='End datetime YYYYMMDDHH (overrides default).')
    parser.add_argument('--delta-hours', type=int, help='Hours between initializations (default 6).')
    parser.add_argument('--input-dir', default=str(DEFAULT_INPUT_DATA_PATH), help='Input directory root.')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DATA_PATH), help='Output directory root.')
    parser.add_argument('--experiment', default=DEFAULT_EXP_NAME, help='PGW experiment label.')
    parser.add_argument('--skip-download', action='store_true', help='Assume inputs already downloaded; skip CDS calls.')
    parser.add_argument('--disable-pgw', action='store_true', help='Disable PGW adjustments inside Aurora prep.')
    args = parser.parse_args()
    require_runtime_dependencies()
    torch.set_grad_enabled(False)
    print(f"CUDA Available: {torch.cuda.is_available()}")

    schedule = build_schedule(
        start=args.start,
        end=args.end,
        delta_hours=args.delta_hours,
        default_start=DEFAULT_FIRST_INIT_TIME,
        default_end=DEFAULT_END_INIT_TIME,
        default_delta_hours=DEFAULT_DELTA_HOURS,
    )
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    experiment_label = args.experiment
    pgw_enabled = DEFAULT_PGW_ENABLED and not args.disable_pgw

    run_forecast(
        schedule,
        input_dir,
        output_dir,
        experiment_label,
        pgw_enabled,
        args.skip_download,
    )
