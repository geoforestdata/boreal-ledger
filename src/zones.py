"""
zones.py

Defines four small representative AOIs (roughly 20-25 km across) for a
cross-biome aboveground carbon comparison. Bounding boxes are illustrative
placeholders centered on well-known locations for each forest type — adjust
if you have more precise site boundaries.

Zones:
  - abitibi: boreal managed forest, Quebec, Canada (near La Sarre/Amos)
  - tapajos: intact tropical rainforest, Para, Brazil (Floresta Nacional
    do Tapajos area — a widely studied carbon-research site)
  - alerce_costero: native temperate forest, Los Rios, Chile (near
    Parque Nacional Alerce Costero)
  - radiata_biobio: even-aged Pinus radiata plantation, Biobio, Chile
    (Arauco/Canete area, heavy industrial plantation coverage)
"""

import ee

ZONES = {
    "abitibi": {
        "label": "Boreal managed forest (Abitibi, Quebec)",
        "bbox": [-78.55, 48.60, -78.30, 48.80],  # [minLon, minLat, maxLon, maxLat]
    },
    "tapajos": {
        "label": "Intact tropical rainforest (Tapajos, Brazil)",
        "bbox": [-55.05, -2.95, -54.80, -2.75],
    },
    "alerce_costero": {
        "label": "Native temperate forest (Alerce Costero, Chile)",
        "bbox": [-73.45, -40.25, -73.20, -40.05],
    },
    "radiata_biobio": {
        "label": "Pinus radiata plantation (Biobio, Chile)",
        "bbox": [-73.40, -37.90, -73.15, -37.70],
    },
}

# Biome type per forest zone, used to select IPCC root:shoot and
# deadwood/litter default factors in carbon_sources.total_forest_carbon.
BIOME_TYPE = {
    "abitibi": "boreal",
    "tapajos": "tropical",
    "alerce_costero": "temperate",
    "radiata_biobio": "plantation",
}

# Wetland zones: illustrative bounding boxes, no forest mask applies (these
# are not forest ecosystems) — used for soil-carbon-dominated comparisons.
# Coordinates below were checked against published site descriptions after
# the first version of this file used unverified placeholders that produced
# implausibly low SOC results (6-14 Mg C/ha, far below real wetland soils).
WETLAND_ZONES = {
    "rocuant_andalien": {
        "label": "Rocuant-Andalien wetland (Biobio, Chile)",
        # Verified: mouth of the Rio Andalien at the Bahia de Concepcion,
        # ~36.74S/73.02W (Wikipedia, "Isla Rocuant"). Original placeholder
        # was offset ~9 km south/west, likely missing the wetland itself.
        "bbox": [-73.08, -36.78, -72.98, -36.70],
    },
    "abitibi_peatland": {
        "label": "Boreal peatland (Abitibi, Quebec)",
        # Located via a PEATGRIDS grid search (src/peatland_finder.py):
        # this candidate showed 2,309 Mg C/ha of full-depth peat carbon,
        # the highest of 9 candidates scanned across the region.
        "bbox": [-79.50, 47.30, -79.25, 47.50],
    },
    "rio_cruces_wetland": {
        "label": "Rio Cruces wetland (Los Rios, Chile)",
        # Verified: Santuario Carlos Anwandter / Rio Cruces Ramsar site,
        # a ~25 km long, ~2 km wide wetland strip roughly between 39.68S
        # and 39.72S, centered near 73.15-73.19W (Wikipedia, Ramsar Sites
        # Information Service #222). Original placeholder was offset west
        # (-73.15 to -73.30), likely missing the river wetland.
        "bbox": [-73.22, -39.75, -73.10, -39.62],
    },
}


def get_wetland_geometry(zone_key: str) -> ee.Geometry:
    """Returns the ee.Geometry.Rectangle for the given wetland zone key."""
    if zone_key not in WETLAND_ZONES:
        raise ValueError(f"Unknown wetland zone '{zone_key}'. Choose from {list(WETLAND_ZONES)}")
    return ee.Geometry.Rectangle(WETLAND_ZONES[zone_key]["bbox"])


def get_wetland_label(zone_key: str) -> str:
    return WETLAND_ZONES[zone_key]["label"]


def get_zone_geometry(zone_key: str) -> ee.Geometry:
    """Returns the ee.Geometry.Rectangle for the given zone key."""
    if zone_key not in ZONES:
        raise ValueError(f"Unknown zone '{zone_key}'. Choose from {list(ZONES)}")
    return ee.Geometry.Rectangle(ZONES[zone_key]["bbox"])


def get_zone_label(zone_key: str) -> str:
    return ZONES[zone_key]["label"]


def point_from_webmercator(x: float, y: float) -> tuple:
    """
    Converts EPSG:3857 (Web Mercator) meters to (lon, lat) in EPSG:4326.
    Use this when a coordinate was copied from a web map (Google Maps,
    QGIS in Web Mercator, etc.) that displays meters instead of degrees.
    """
    import math

    R = 20037508.34
    lon = (x / R) * 180
    lat_deg = (y / R) * 180
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat_deg * math.pi / 180)) - math.pi / 2)
    return lon, lat


def get_point_buffer_zone(lon: float, lat: float, radius_m: float = 500) -> ee.Geometry:
    """
    Returns a circular buffer of radius_m around (lon, lat) in EPSG:4326.
    Use this for point-based site comparisons where every zone must have
    IDENTICAL area regardless of terrain shape (a fixed-radius circle
    guarantees equal area across all zones, unlike a bounding box that can
    end up mostly water/non-forest depending on where it lands).
    """
    return ee.Geometry.Point([lon, lat]).buffer(radius_m)

