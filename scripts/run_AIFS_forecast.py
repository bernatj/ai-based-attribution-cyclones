# run_AIFS_forecast
# This notebook runs ECMWF's aifs-single-v1 data-driven model, using ECMWF's [open data](https://www.ecmwf.int/en/forecasts/datasets/open-data) dataset and the [anemoi-inference](https://anemoi-inference.readthedocs.io/en/latest/apis/level1.html) package.


import argparse
import datetime
import copy
import os
from pathlib import Path
from collections import defaultdict
import sys
import xarray as xr
from tqdm import tqdm  # for progress bar

import numpy as np

try:
    import cdsapi
    import earthkit.regrid as ekr
    from anemoi.inference.outputs.printer import print_state
    from anemoi.inference.runners.simple import SimpleRunner
except ModuleNotFoundError:
    cdsapi = None
    ekr = None
    print_state = None
    SimpleRunner = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.paths import aifs_root


####################### Configuration ################
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]
PARAM_SOIL = ["swvl", "stl"]
PARAM_PL = ["z", "t", "u", "v", "w", "q"]
STATIC_PARAMS = ["lsm", "z", "slor", "sdor"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS = [1, 2]
DEFAULT_FIRST_INIT_TIME = datetime.datetime(2025, 11, 8, 0)
DEFAULT_END_INIT_TIME = datetime.datetime(2025, 11, 15, 0)
DEFAULT_DELTA_HOURS = 6
DEFAULT_OUTPUT_DIR = aifs_root()
RUN_HOURS = 10 * 24
PGW_OPT = True
EXPER = 'PGW_multimodel_v1'  # Name of the PGW experiment to use
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
FIRST_INIT_TIME = DEFAULT_FIRST_INIT_TIME
END_INIT_TIME = DEFAULT_END_INIT_TIME
DELTA_HOURS = DEFAULT_DELTA_HOURS
SKIP_DOWNLOAD = False


def _parse_datetime(value, default):
    if value:
        return datetime.datetime.strptime(value, '%Y%m%d%H')
    return default
SAVE_VAR_LIST = PARAM_SFC + ['swvl1', 'swvl2', 'stl1', 'stl2', 'tp', 'cp', '100u', '100v'] + ['q_850', 'v_850', 'u_850', 't_850', 't_500', 'z_500', 'u_300', 'v_300']
####################################################


def require_runtime_dependencies() -> None:
    missing = []
    if cdsapi is None:
        missing.append('cdsapi')
    if ekr is None:
        missing.append('earthkit-regrid')
    if SimpleRunner is None or print_state is None:
        missing.append('anemoi-inference')
    if missing:
        joined = ', '.join(missing)
        raise SystemExit(
            f"Missing AIFS runtime dependencies: {joined}. Activate the AIFS environment before running this script."
        )


# ---------------------------
# Helper Functions
# ---------------------------
def generate_init_times(start, end, delta_hours):
    times = []
    current = start
    while current <= end:
        times.append(current)
        current += datetime.timedelta(hours=delta_hours)
    return times

def download_static_file():
    """Download static variables that don't change with time"""
    require_runtime_dependencies()
    client = cdsapi.Client()
    static_file = OUTPUT_DIR / 'input' / "static.nc"
    if not static_file.exists():
        print("Downloading static data...")
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "land_sea_mask",                    # lsm
                    "geopotential",                     # z
                    "slope_of_subgridscale_orography",  # slor
                    "standard_deviation_of_orography"   # sdor
                ],
                "year": "2000",  # Any year works for static data
                "month": "01",
                "day": "01",
                "time": "00:00",
                "format": "netcdf",
            },
            str(static_file))
        print(f"Static data saved to {static_file}")
    return static_file

def download_aifs_init_data(DATE):
    """Download AIFS initialization data for current and previous timestep"""
    require_runtime_dependencies()
    client = cdsapi.Client()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Get both current and previous time step
    for date in [DATE - datetime.timedelta(hours=6), DATE]:
        yyyymmddhh = date.strftime('%Y%m%d%H')
        date_dir = OUTPUT_DIR / 'input' / yyyymmddhh
        date_dir.mkdir(exist_ok=True)

        # 1. Download surface data (excluding static variables)
        print(f"Downloading surface initialization data for {yyyymmddhh}...")
        surface_file = date_dir / f"{yyyymmddhh}-surface.nc"
        if not surface_file.exists():
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "10m_u_component_of_wind",  # 10u
                        "10m_v_component_of_wind",  # 10v
                        "2m_dewpoint_temperature",  # 2d
                        "2m_temperature",           # 2t
                        "mean_sea_level_pressure",  # msl
                        "skin_temperature",         # skt
                        "surface_pressure",         # sp
                        "total_column_water"        # tcw
                    ],
                    "year": date.year,
                    "month": date.month,
                    "day": date.day,
                    "time": f"{date.hour:02d}:00",
                    "format": "netcdf",
                },
                str(surface_file))
            print(f"Surface data for {yyyymmddhh} saved to {surface_file}")
        
        # 2. Download soil data
        print(f"Downloading soil data for {yyyymmddhh}...")
        soil_file = date_dir / f"{yyyymmddhh}-soil.nc"
        if not soil_file.exists():
            client.retrieve(
                #"reanalysis-era5-land"
                 "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "volumetric_soil_water_layer_1",
                        "volumetric_soil_water_layer_2",
                        "soil_temperature_level_1",
                        "soil_temperature_level_2"
                    ],
                    "year": date.year,
                    "month": date.month,
                    "day": date.day,
                    "time": f"{date.hour:02d}:00",
                    "area": [90, -180, -90, 180],  # Full globe
                    "grid": [0.25, 0.25],           # 0.25°x0.25° resolution
                    "format": "netcdf",  # <-- change this
                },
                str(soil_file))
            print(f"Soil data for {yyyymmddhh} saved to {soil_file}")
        
        # 3. Download pressure level data
        print(f"Downloading pressure level data for {yyyymmddhh}...")
        pl_file = date_dir / f"{yyyymmddhh}-pressure-levels.nc"
        if not pl_file.exists():
            client.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "geopotential",         # gh
                        "temperature",          # t
                        "u_component_of_wind",  # u
                        "v_component_of_wind",  # v
                        "vertical_velocity",    # w
                        "specific_humidity"     # q
                    ],
                    "pressure_level": LEVELS,
                    "year": date.year,
                    "month": date.month,
                    "day": date.day,
                    "time": f"{date.hour:02d}:00",
                    "format": "netcdf",
                },
                str(pl_file))
            print(f"Pressure level data for {yyyymmddhh} saved to {pl_file}")


def prepare_aifs_input(DATE):
    """Prepare input data (static + dynamic) interpolated and stacked for AIFS"""
    require_runtime_dependencies()
    from collections import defaultdict

    fields = defaultdict(list)

    # 1. Load and interpolate static fields, duplicate for both timesteps
    static_ds = xr.open_dataset(OUTPUT_DIR / 'input' / "static.nc")

    for param in STATIC_PARAMS:
        arr = static_ds[param].values
        interp = ekr.interpolate(arr, {"grid": (0.25, 0.25)}, {"grid": "N320"})
        fields[param].extend([interp, interp])  # Add twice (for both time steps)

    # 2. Loop over two timesteps: DATE - 6h and DATEye
    for i, date in enumerate([DATE - datetime.timedelta(hours=6), DATE]):
        yyyymmddhh = date.strftime('%Y%m%d%H')
        date_dir = OUTPUT_DIR / 'input' / yyyymmddhh

        # --- Surface data ---
        sfc_file = f"{yyyymmddhh}-surface.nc"
        if PGW_OPT:
            sfc_file = f"{yyyymmddhh}-surface-{EXPER}.nc"
        with xr.open_dataset(date_dir / sfc_file) as ds:
            var_map = {
                "10u": "u10",
                "10v": "v10",
                "2d": "d2m",
                "2t": "t2m",
                "msl": "msl",
                "skt": "skt",
                "sp": "sp",
                "tcw": "tcw"
            }
            for param in PARAM_SFC:
                arr = ds[var_map[param]].values
                interp = ekr.interpolate(arr, {"grid": (0.25, 0.25)}, {"grid": "N320"})
                fields[param].append(interp)

        # --- Soil data ---
        with xr.open_dataset(date_dir / f"{yyyymmddhh}-soil.nc") as ds:
            for level in SOIL_LEVELS:
                for param in PARAM_SOIL:
                    values = ds[f"{param}{level}"].values                    
                    values = np.roll(values, -values.shape[2] // 2, axis=2) #these were in a [-180,180 range]                    
                    interp = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"}) 

                    #substitue over ocean gridpoits for skt (which is aproximatelly SSTs). 
                    #needed for ERALand and not for ERA5
                    # if param == 'stl':                
                    #    interp = np.where(np.isnan(interp), fields["skt"][i],interp)
                    #elif param == 'swvl':
                    #    interp = np.where(np.isnan(interp), 0.0 ,interp)

                    fields[f"{param}{level}"].append(interp)


        # --- Pressure level data ---
        p_file = f"{yyyymmddhh}-pressure-levels.nc"
        if PGW_OPT:
            p_file = f"{yyyymmddhh}-pressure-levels-{EXPER}.nc"
        with xr.open_dataset(date_dir / p_file) as ds:
            for param in PARAM_PL:
                for level in LEVELS:
                    arr = ds[param].sel(pressure_level=level).values
                    interp = ekr.interpolate(arr, {"grid": (0.25, 0.25)}, {"grid": "N320"})
                    key = f"{param}_{level}"
                    fields[key].append(interp)

    # 3. Stack two time steps: shape becomes (2, 542080)
    fields = {k: np.stack(v) for k, v in fields.items()}

    return {
        "date": DATE,
        "fields": fields
    }

def save_variables_separately(states, yyyymmddhh, output_dir='.'):
    """Save each variable interpolated to 0.25° regular grid"""
    require_runtime_dependencies()
    
    # Define target grid (0.25° regular lat-lon)
    target_grid = {"grid": (0.25, 0.25)}  # 0.25° resolution
    source_grid = {"grid": "N320"}  # AIFS native grid
    
    # Get all variable names from first state
    var_names = list(states[0]['fields'].keys())
    
    # Create time coordinate from the dates
    times = [datetime.strptime(s['date'], '%Y-%m-%d %H:%M:%S') if isinstance(s['date'], str) 
             else s['date'] for s in states]
       
    # Target latitude and longitude arrays (0.25° resolution)
    target_lats = np.arange(90, -90.25, -0.25)
    target_lons = np.arange(0, 360, 0.25)  # Using 0-360 for consistency with ECMWF
    
    for var_name in tqdm(SAVE_VAR_LIST, desc="Processing variables"):
        # Initialize array for this variable (time, lat, lon)
        var_data = np.zeros((len(times), len(target_lats), len(target_lons)))
        
        for i, state in enumerate(states):
            # Get original values
            values = state['fields'][var_name]
            
            # Interpolate using earthkit
            interpolated = ekr.interpolate(values, source_grid, target_grid)

            # Reshape to 2D (assuming earthkit returns flattened array)
            var_data[i, :, :] = interpolated.reshape(len(target_lats), len(target_lons))
        
        # Create xarray Dataset
        ds = xr.Dataset(
            {var_name: (['time', 'lat', 'lon'], var_data)},
            coords={
                'time': times,
                'lat': target_lats,
                'lon': target_lons
            }
        )
        
        # Add CF-compliant metadata
        ds[var_name].attrs = {
            'units': 'unknown', 
            'long_name': var_name,
            'grid_mapping': 'latitude_longitude'
        }
        ds.lat.attrs = {'units': 'degrees_north', 'axis': 'Y'}
        ds.lon.attrs = {'units': 'degrees_east', 'axis': 'X'}
        ds.time.attrs = {'long_name': 'time', 'axis': 'T'}
        
        # Encoding for compression
        encoding = {
            var_name: {
                'zlib': True,
                'complevel': 4,
                'dtype': 'float32'
            },
            'time': {'dtype': 'float64'}  # More precise time storage
        }
        
        # Save to NetCDF
        filename = f"{output_dir}/{var_name}_aifs_{yyyymmddhh}.nc"
        if PGW_OPT:
            filename = f"{output_dir}/{var_name}_aifs_{EXPER}_{yyyymmddhh}.nc"
        ds.to_netcdf(filename, encoding=encoding)
        print(f"Saved {filename}")

# ---------------------------
# Main Forecast Loop
# ---------------------------
def run_forecast():
    require_runtime_dependencies()

    if SKIP_DOWNLOAD:
        static_file = OUTPUT_DIR / 'input' / "static.nc"
    else:
        static_file = download_static_file()

    #Load the Model and Run the Forecast
    print("Initializing model...")

    checkpoint = {"huggingface":"ecmwf/aifs-single-1.0"}
    runner = SimpleRunner(checkpoint, device=os.getenv('CYCLONE_AIFS_DEVICE', 'cuda:0'))

    init_dates = generate_init_times(FIRST_INIT_TIME, END_INIT_TIME, DELTA_HOURS)

    for DATE in init_dates:
        print(f"\nProcessing forecast for: {DATE}")

        if not SKIP_DOWNLOAD:
            download_aifs_init_data(DATE)

        #Prepare AIFS input
        aifs_input = prepare_aifs_input(DATE)
        print("AIFS input data prepared successfully!")
        print(f"Contains {len(aifs_input['fields'])} fields")
        input_state = dict(date=DATE, fields=aifs_input['fields'])
    
        ## Execute and Collect all forecasts
        states = []
        for state in runner.run(input_state=input_state, lead_time=RUN_HOURS):
            print_state(state)
            states.append(copy.deepcopy(state))

        ##Save the forecasts in one file per variable
        yyyymmddhh =DATE.strftime('%Y%m%d%H')
        date_dir = OUTPUT_DIR / 'fcst' / yyyymmddhh
        date_dir.mkdir(exist_ok=True)
        save_variables_separately(states, yyyymmddhh, date_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the AIFS forecast workflow.')
    parser.add_argument('--start', help='Start datetime YYYYMMDDHH (overrides env/default).')
    parser.add_argument('--end', help='End datetime YYYYMMDDHH (overrides env/default).')
    parser.add_argument('--delta-hours', type=int, help='Hour step between inits (default 6).')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='Root output directory.')
    parser.add_argument('--experiment', default='PGW_multimodel_v1', help='PGW experiment label suffix.')
    parser.add_argument('--skip-download', action='store_true', help='Assume inputs already downloaded; skip CDS calls.')
    parser.add_argument('--disable-pgw', action='store_true', help='Disable PGW adjustments inside the runner.')
    return parser.parse_args()


def main():
    global FIRST_INIT_TIME, END_INIT_TIME, DELTA_HOURS, OUTPUT_DIR, EXPER, PGW_OPT, SKIP_DOWNLOAD
    args = parse_args()
    FIRST_INIT_TIME = _parse_datetime(args.start, DEFAULT_FIRST_INIT_TIME)
    END_INIT_TIME = _parse_datetime(args.end, DEFAULT_END_INIT_TIME)
    DELTA_HOURS = args.delta_hours if args.delta_hours else DEFAULT_DELTA_HOURS
    OUTPUT_DIR = Path(args.output_dir)
    EXPER = args.experiment
    PGW_OPT = not args.disable_pgw
    SKIP_DOWNLOAD = args.skip_download
    run_forecast()


if __name__ == "__main__":
    main()
