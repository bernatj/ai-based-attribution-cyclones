Minimal CMIP6 multimodel-mean delta bundle for the supported PGW workflow.

This directory is intentionally small. It only contains the `multimodel_mean_10models_v1`
NetCDF files needed by the four supported models:

- `tas/`
- `prw/`
- `ta/`
- `hur/`
- `hus/`
- `tos/`

Expected filenames:

- `tas/tas_multimodel_mean_10models_v1.nc`
- `prw/prw_multimodel_mean_10models_v1.nc`
- `ta/ta_multimodel_mean_10models_v1.nc`
- `hur/hur_multimodel_mean_10models_v1.nc`
- `hus/hus_multimodel_mean_10models_v1.nc`
- `tos/tos_multimodel_mean_10models_v1.nc`

If you move this folder outside the repo, set:

```bash
export CYCLONE_CMIP6_DELTA_ROOT=/path/to/cmip6_deltas
```
