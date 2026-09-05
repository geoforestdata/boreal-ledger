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
WETLAND_ZONES = {
    "rocuant_andalien": {
        "label": "Rocuant-Andalien wetland (Biobio, Chile)",
        "bbox": [-73.15, -36.87, -73.00, -36.77],
    },
    "abitibi_peatland": {
        "label": "Boreal peatland (Abitibi, Quebec)",
        "bbox": [-79.30, 48.70, -79.10, 48.85],
    },
    "rio_cruces_wetland": {
        "label": "Rio Cruces wetland (Los Rios, Chile)",
        "bbox": [-73.30, -39.82, -73.15, -39.68],
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

