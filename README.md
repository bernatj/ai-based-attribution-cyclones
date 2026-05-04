# ai-based-attribution-cyclones

Repository accompanying the paper: "Forecast-based Attribution of Extratropical Cyclones Using AI Weather Models"

Minimal reproduction repo for the four AI weather prediction models used in the paper:

- `pangu`
- `fcnv2`
- `aurora`
- `aifs`

The repo is intentionally manual. Run the scripts directly, in order, and then open the notebooks.

## Repo Layout

- `scripts/`: model download, PGW, and forecast runners
- `notebooks/extratropical_storm_multimodel_forecast_skill_attribution.ipynb`: main multimodel notebook for factual and PGW diagnostics
- `notebooks/extratropical_storm_AIFS_precip_attribution.ipynb`: AIFS precipitation notebook
- `cmip6_deltas/`: minimal CMIP6 multimodel-mean delta bundle for the supported PGW workflow
- `envs/`: starter conda environments, one per model

Optional helper:

- `scripts/download_ifs_data.py`: download IFS reference forecasts from ECMWF open data directly into the notebook layout
- `scripts/convert_ifs_grib_to_netcdf.py`: convert a single IFS GRIB file to NetCDF

## Environment Notes

The exact environment for each AIWP model can depend on the hardware used.

In practice, the main sources of variation are:

- GPU model and available VRAM
- CUDA version and driver version
- `torch` build
- model-specific packages such as `earth2mip`, `ddf`, `aurora`, and `anemoi-*`

The YAML files in `envs/` are starting points, not strict lockfiles. You may need to adjust `torch`, CUDA, or model-package installation to match your machine.

Starter env files:

- `envs/pangu.yml`
- `envs/fcnv2.yml`
- `envs/aurora.yml`
- `envs/aifs.yml`
- `envs/notebooks.yml`

Create the starter envs with:

```bash
conda env create -f envs/pangu.yml
conda env create -f envs/fcnv2.yml
conda env create -f envs/aurora.yml
conda env create -f envs/aifs.yml
conda env create -f envs/notebooks.yml
```

You only need the envs for the models you plan to run.
The `notebooks` env is separate and only intended for analysis and plotting.
It also includes the optional ECMWF open-data client used by `scripts/download_ifs_data.py`.

Useful environment variables:

- `CYCLONE_DATA_ROOT`: base directory for forecast/input data. Default: `/home/bernatj/Data`
- `CYCLONE_AI_FORECAST_ROOT`: shared Earth2MIP input/output root for `pangu` and `fcnv2`
- `CYCLONE_AURORA_ROOT`: Aurora input/output root
- `CYCLONE_AIFS_ROOT`: AIFS input/output root
- `CYCLONE_IFS_ROOT`: optional IFS reference root
- `CYCLONE_ERA5_PRECIP_ROOT`: ERA5 precipitation root used by the AIFS notebook
- `CYCLONE_CMIP6_DELTA_ROOT`: CMIP6 multimodel delta directory. Default: repo-local `cmip6_deltas/` if present
- `DDF_PATH`: optional path to a local `ddf` checkout for Earth2MIP runs
- `EARTH2MIP_PATH`: optional path to a local `earth2mip` checkout
- `CYCLONE_AURORA_DEVICE`: optional torch device for Aurora. Default: `cuda:0`
- `CYCLONE_AIFS_DEVICE`: optional torch device for AIFS. Default: `cuda:0`

Example:

```bash
export CYCLONE_DATA_ROOT=/path/to/Data
export CYCLONE_CMIP6_DELTA_ROOT=/path/to/cmip6_deltas
export DDF_PATH=/path/to/ddf
export EARTH2MIP_PATH=/path/to/earth2mip
```

If you keep the bundled `cmip6_deltas/` directory inside the repo, you do not need to set `CYCLONE_CMIP6_DELTA_ROOT`.

## Factual Workflow

The downloader for `pangu` and `fcnv2` is shared.

### 1. Download shared Earth2MIP initial conditions

```bash
conda activate pangu
python scripts/download_inidata_for_aimodel.py \
  --models pangu,fcnv2 \
  --start 2025110700 \
  --end 2025111400 \
  --delta-hours 6
```

### 2. Run Pangu factual forecasts

```bash
conda activate pangu
python scripts/run_aimodel_general.py \
  --ai-model pangu \
  --model-name pangu \
  --file-format grib \
  --start 2025110700 \
  --end 2025111400 \
  --delta-hours 6
```

### 3. Run FCNV2 factual forecasts

```bash
conda activate fcnv2
python scripts/run_aimodel_general.py \
  --ai-model fcnv2 \
  --model-name fcnv2_sm \
  --file-format grib \
  --start 2025110700 \
  --end 2025111400 \
  --delta-hours 6
```

### 4. Download Aurora inputs

```bash
conda activate aurora
python scripts/download_aurora_data.py \
  --start 2025110700 \
  --end 2025111400 \
  --delta-hours 6
```

### 5. Run Aurora factual forecasts

```bash
conda activate aurora
python scripts/run_aurora_forecast.py \
  --start 2025110700 \
  --end 2025111400 \
  --delta-hours 6 \
  --skip-download \
  --disable-pgw
```

### 6. Download AIFS inputs

```bash
conda activate aifs
python scripts/download_aifs_data.py \
  --start 2025110700 \
  --end 2025111400 \
  --delta-hours 6
```

### 7. Run AIFS factual forecasts

```bash
conda activate aifs
python scripts/run_AIFS_forecast.py \
  --start 2025110700 \
  --end 2025111400 \
  --delta-hours 6 \
  --skip-download \
  --disable-pgw
```

### 8. Open the notebook

Open:

- `notebooks/extratropical_storm_multimodel_forecast_skill_attribution.ipynb`

Use:

```bash
conda activate notebooks
jupyter lab
```

The main notebook is centered on the four paper AI models for the PGW attribution workflow.
For factual evaluation plots, it can also include an optional `IFS` reference if IFS files are available under `CYCLONE_IFS_ROOT`.

## PGW Workflow

Use one experiment label consistently. The notebooks currently expect:

```bash
export PGW_EXPERIMENT=PGW_multimodel_v1
```

For `pangu` and `fcnv2`, use `--experiment "_${PGW_EXPERIMENT}"`.

### 1. Reuse the factual downloads or rerun them

For `pangu` and `fcnv2`, reuse the GRIB downloads from the factual workflow.

### 2. Apply PGW deltas and rerun Pangu

```bash
conda activate pangu
python scripts/apply_pgw_deltas_to_inicon_cmip6.py \
  --ai-model pangu \
  --grib \
  --start 2025110700 \
  --end 2025111400 \
  --delta-hours 6 \
  --path-delta-mm "$CYCLONE_CMIP6_DELTA_ROOT" \
  --delta-file-name multimodel_mean_10models_v1 \
  --exp-name "$PGW_EXPERIMENT"

python scripts/run_aimodel_general.py \
  --ai-model pangu \
  --model-name pangu \
  --file-format netcdf \
  --experiment "_${PGW_EXPERIMENT}" \
  --start 2025110700 \
  --end 2025111400 \
  --delta-hours 6
```

### 3. Apply PGW deltas and rerun FCNV2

```bash
conda activate fcnv2
python scripts/apply_pgw_deltas_to_inicon_cmip6.py \
  --ai-model fcnv2 \
  --grib \
  --start 2025110700 \
  --end 2025111400 \
  --delta-hours 6 \
  --path-delta-mm "$CYCLONE_CMIP6_DELTA_ROOT" \
  --delta-file-name multimodel_mean_10models_v1 \
  --exp-name "$PGW_EXPERIMENT"

python scripts/run_aimodel_general.py \
  --ai-model fcnv2 \
  --model-name fcnv2_sm \
  --file-format netcdf \
  --experiment "_${PGW_EXPERIMENT}" \
  --start 2025110700 \
  --end 2025111400 \
  --delta-hours 6
```

### 4. Apply PGW deltas and rerun Aurora

```bash
conda activate aurora
python scripts/apply_pgw_deltas_to_inicon_cmip6.py \
  --ai-model aurora \
  --start 2025110700 \
  --end 2025111400 \
  --delta-hours 6 \
  --path-delta-mm "$CYCLONE_CMIP6_DELTA_ROOT" \
  --delta-file-name multimodel_mean_10models_v1 \
  --exp-name "$PGW_EXPERIMENT"

python scripts/run_aurora_forecast.py \
  --start 2025110700 \
  --end 2025111400 \
  --delta-hours 6 \
  --skip-download \
  --experiment "$PGW_EXPERIMENT"
```

### 5. Apply PGW deltas and rerun AIFS

```bash
conda activate aifs
python scripts/apply_pgw_deltas_to_inicon_cmip6.py \
  --ai-model aifs \
  --start 2025110700 \
  --end 2025111400 \
  --delta-hours 6 \
  --path-delta-mm "$CYCLONE_CMIP6_DELTA_ROOT" \
  --delta-file-name multimodel_mean_10models_v1 \
  --modify-skt \
  --exp-name "$PGW_EXPERIMENT"

python scripts/run_AIFS_forecast.py \
  --start 2025110700 \
  --end 2025111400 \
  --delta-hours 6 \
  --skip-download \
  --experiment "$PGW_EXPERIMENT"
```

### 6. Open the notebook(s)

Open either:

- `notebooks/extratropical_storm_multimodel_forecast_skill_attribution.ipynb`
- `notebooks/extratropical_storm_AIFS_precip_attribution.ipynb`

Set `storm_name` in the notebook configuration cell before running the analysis.

## Notes

- Factual `pangu` and `fcnv2` runs read the original GRIB downloads.
- PGW `pangu` and `fcnv2` runs read the PGW NetCDF files produced by `apply_pgw_deltas_to_inicon_cmip6.py`.
- The PGW experiment label used in the reruns must match the label expected by the notebooks.
- The notebooks use env-backed data roots and repo-local imports.
- The bundled `cmip6_deltas/` folder contains only the multimodel-mean NetCDF files needed by the supported four-model PGW workflow, so it can be uploaded or shared independently from the rest of the data tree.

## Optional IFS Route

IFS is not part of the default four-model PGW workflow, but the main notebook can read IFS reference files for factual comparison plots.
The supported downloader here is ECMWF open data only.

Use the `notebooks` environment for IFS conversion and analysis.

Preferred path:

```bash
conda activate notebooks
python scripts/download_ifs_data.py \
  --start 2025110700 \
  --end 2025111400 \
  --delta-hours 12 \
  --wind-level 850
```

This writes one NetCDF file per variable and init time under `CYCLONE_IFS_ROOT`.
With `--wind-level 850`, the downloader matches the current notebook default `use_850_winds = True`.

Expected output layout:

```text
$CYCLONE_IFS_ROOT/YYYYMMDDHH/
  u_ifs_YYYYMMDDHH.nc
  v_ifs_YYYYMMDDHH.nc
  msl_ifs_YYYYMMDDHH.nc
  q_ifs_YYYYMMDDHH.nc
```

If your IFS files are still in GRIB format, convert them first:

```bash
conda activate notebooks
python scripts/convert_ifs_grib_to_netcdf.py input.grib output.nc
```

The loader in `scripts/utils.py` already supports `model='ifs'`, including coordinate normalization for common IFS NetCDF/GRIB exports.

This optional IFS route is factual-only in the current repo. There is no bundled PGW or automated IFS workflow here.
