"""
09_agb_crosscheck_ntems.py

Diagnostic access check for the Canadian Forest Service / NTEMS
"Forest Total Biomass (2022)" product.

The official 2022 layer is available as a public 30 m GeoTIFF inside:
  https://opendata.nfis.org/downloads/forest_change/CA_total_biomass_2022.zip

The WMS layer documents the variable as aboveground biomass, but it is not
queryable. The public WCS endpoint inspected during this audit did not expose
the biomass coverage, only forest-change coverages. Because the ZIP is about
10.25 GB, this script does not download it automatically.

If a local unzipped GeoTIFF is available, run:
  python analysis/09_agb_crosscheck_ntems.py /path/to/CA_total_biomass_2022.tif
"""

from __future__ import annotations

import sys
from pathlib import Path


AOI_BBOX_WGS84 = (-79.61448860036347, 48.32451136817797, -79.47908860036347, 48.414511368177976)
ZIP_URL = "https://opendata.nfis.org/downloads/forest_change/CA_total_biomass_2022.zip"
WMS_URL = (
    "https://opendata.nfis.org/mapserver/cgi-bin/wms_change.cgi?"
    "SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0&layers=CA_total_biomass_2022"
)
WCS_URL = "https://opendata.nfis.org/mapserver/cgi-bin/wcs_change.cgi?SERVICE=WCS&REQUEST=GetCapabilities&VERSION=2.0.1"


def mean_from_local_geotiff(path: Path) -> tuple[float, int]:
    import numpy as np
    from osgeo import gdal

    ds = gdal.Open(str(path))
    if ds is None:
        raise RuntimeError(f"Could not open raster: {path}")

    clipped = gdal.Warp(
        "",
        ds,
        format="MEM",
        dstSRS="EPSG:4326",
        outputBounds=AOI_BBOX_WGS84,
        outputBoundsSRS="EPSG:4326",
        xRes=0.00027,
        yRes=0.00027,
        resampleAlg="near",
    )
    if clipped is None:
        raise RuntimeError("GDAL could not clip the AOI from the NTEMS raster.")

    band = clipped.GetRasterBand(1)
    arr = band.ReadAsArray().astype(float)
    nodata = band.GetNoDataValue()
    mask = np.isfinite(arr)
    if nodata is not None:
        mask &= arr != nodata
    mask &= arr >= 0

    values = arr[mask]
    if values.size == 0:
        raise RuntimeError("No valid NTEMS biomass pixels found in the AOI clip.")
    return float(values.mean()), int(values.size)


def main() -> int:
    print("=== NTEMS 2022 Forest Total Biomass access diagnostic ===\n")
    print("Official product: Forest Total Biomass (2022)")
    print("Layer: CA_total_biomass_2022")
    print(f"Download ZIP: {ZIP_URL}")
    print(f"WMS capabilities: {WMS_URL}")
    print(f"WCS capabilities: {WCS_URL}")
    print(
        "Definition from the official WMS abstract: 30 m forest-structure estimates "
        "including aboveground biomass, derived from airborne lidar and Landsat "
        "composites. The broader 2015 biomass layer defines total aboveground "
        "biomass as the sum of individual-tree aboveground biomass per hectare."
    )
    print(
        "Access note: the WMS layer is advertised with queryable=0, and the WCS "
        "capabilities available during this audit did not advertise the 2022 "
        "biomass coverage. The public ZIP is about 10.25 GB, so it is not "
        "downloaded automatically.\n"
    )

    if len(sys.argv) < 2:
        print("NTEMS result: not computed in this run.")
        print("Reason: provide a local unzipped CA_total_biomass_2022.tif to compute the AOI mean exactly.")
        return 0

    raster_path = Path(sys.argv[1]).expanduser()
    mean_biomass, pixel_count = mean_from_local_geotiff(raster_path)
    print(f"Local raster: {raster_path}")
    print(f"Valid AOI pixels: {pixel_count:,}")
    print(f"Mean aboveground biomass: {mean_biomass:.2f} Mg biomass/ha")
    print(f"Mean aboveground carbon at 0.47: {mean_biomass * 0.47:.2f} Mg C/ha")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
