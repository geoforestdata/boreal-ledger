"""
gee_utils.py

Reusable functions for extracting annual cloud-masked Sentinel-2 composites
over a managed forest area and computing spectral indices used for change
detection (NDVI, NBR). Imported from the notebooks in /notebooks.
"""

import ee


def init_ee(project: str) -> None:
    """Initializes the Earth Engine API with the given cloud project."""
    ee.Initialize(project=project)


def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Masks clouds and cloud shadows using the Scene Classification Layer."""
    scl = image.select("SCL")
    # SCL classes excluded: 3 = cloud shadow, 8/9 = cloud medium/high prob,
    # 10 = thin cirrus, 11 = snow/ice.
    mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return image.updateMask(mask).divide(10000).copyProperties(
        image, ["system:time_start"]
    )


def add_indices(image: ee.Image) -> ee.Image:
    """Adds NDVI and NBR bands to a Sentinel-2 SR image."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    nbr = image.normalizedDifference(["B8", "B12"]).rename("NBR")
    return image.addBands([ndvi, nbr])


def annual_composite(year: int, aoi: ee.Geometry,
                      cloud_pct_max: int = 40) -> ee.Image:
    """
    Builds a cloud-masked median composite for the given year, restricted to
    the leaf-on window (June-September) to reduce phenological noise between
    years.
    """
    start = ee.Date.fromYMD(year, 6, 1)
    end = ee.Date.fromYMD(year, 9, 30)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct_max))
        .map(mask_s2_clouds)
        .map(add_indices)
    )

    composite = collection.median().clip(aoi)
    return composite.set({"year": year, "system:time_start": start.millis()})


def build_annual_stack(years: range, aoi: ee.Geometry) -> ee.ImageCollection:
    """Returns an ImageCollection of annual NDVI/NBR composites."""
    images = [
        annual_composite(y, aoi).select(["NDVI", "NBR"]).set("year", y)
        for y in years
    ]
    return ee.ImageCollection(images)


def export_annual_composites(
    years: range,
    aoi: ee.Geometry,
    drive_folder: str = "gee_quebec_forest_change",
    scale: int = 10,
):
    """Kicks off one Drive export task per year for NDVI + NBR bands."""
    tasks = []
    for year in years:
        image = annual_composite(year, aoi).select(["NDVI", "NBR"])
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=f"s2_composite_{year}",
            folder=drive_folder,
            fileNamePrefix=f"s2_ndvi_nbr_{year}",
            region=aoi,
            scale=scale,
            crs="EPSG:32198",  # NAD83 / Quebec Lambert, matches MRNF datasets
            maxPixels=1e13,
        )
        task.start()
        tasks.append(task)
        print(f"Started export task for {year}: {task.id}")
    return tasks

