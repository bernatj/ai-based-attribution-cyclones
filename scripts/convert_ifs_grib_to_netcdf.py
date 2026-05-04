"""Convert a single IFS GRIB file to NetCDF."""

from __future__ import annotations

import argparse
from pathlib import Path


def convert_grib_to_netcdf(input_grib: Path, output_netcdf: Path) -> None:
    import xarray as xr

    output_netcdf.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(input_grib, engine='cfgrib')
    ds.to_netcdf(output_netcdf)
    print(f"Converted {input_grib} -> {output_netcdf}")


def parse_args():
    parser = argparse.ArgumentParser(description='Convert one IFS GRIB file to NetCDF.')
    parser.add_argument('input_grib', help='Input GRIB file path.')
    parser.add_argument('output_netcdf', help='Output NetCDF file path.')
    return parser.parse_args()


def main():
    args = parse_args()
    convert_grib_to_netcdf(Path(args.input_grib), Path(args.output_netcdf))


if __name__ == '__main__':
    main()
