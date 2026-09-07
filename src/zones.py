"""
Abitibi study-area geometry used by the boreal-ledger analyses.

The project works on a single managed boreal landscape in
Abitibi-Temiscamingue, Quebec. The rectangle is an analysis AOI, not an
administrative unit. It was selected after checking that Hansen
treecover2000 > 50 covers most of the box and that the geometry avoids the
lake-dominated regional extent.
"""

import math

import ee


ABITIBI_ZONE_LABEL = "Managed boreal landscape (Abitibi-Temiscamingue, Quebec)"
ABITIBI_10K_CENTER_WEBMERCATOR = (-8855108, 6168550)
ABITIBI_10K_HALF_WIDTH_LON = 0.0677
ABITIBI_10K_HALF_WIDTH_LAT = 0.0450


def point_from_webmercator(x: float, y: float) -> tuple:
    """
    Converts EPSG:3857 Web Mercator meters to (lon, lat) in EPSG:4326.
    """
    radius = 20037508.34
    lon = (x / radius) * 180
    lat_deg = (y / radius) * 180
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat_deg * math.pi / 180)) - math.pi / 2)
    return lon, lat


def get_abitibi_analysis_zone() -> ee.Geometry:
    """Returns the verified ~10,000 ha Abitibi analysis rectangle."""
    lon, lat = point_from_webmercator(*ABITIBI_10K_CENTER_WEBMERCATOR)
    return ee.Geometry.Rectangle([
        lon - ABITIBI_10K_HALF_WIDTH_LON,
        lat - ABITIBI_10K_HALF_WIDTH_LAT,
        lon + ABITIBI_10K_HALF_WIDTH_LON,
        lat + ABITIBI_10K_HALF_WIDTH_LAT,
    ])
