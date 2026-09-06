"""
08_agb_crosscheck_mrnf.py

Area-weighted AGB cross-check using the official Quebec MRNF layer
"Biomasse et carbone forestiers du Quebec meridional" exposed by WFS.

Layers used:
  bio_arbv_tot  -> b_arbv_tot, aboveground live-tree biomass, dry t/ha
  carb_arbv_tot -> c_arbv_tot, aboveground live-tree carbon, t C/ha

Official layer abstract: all species and all diameter classes; aboveground
live-tree components include stump, stem, branches, foliage, wood, bark,
leaves or needles; roots are excluded. Values come from forest compilations
by stand ("compilations forestieres par peuplement").

Run: python 08_agb_crosscheck_mrnf.py
"""

import sys
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import transform

from zones import get_abitibi_analysis_zone, point_from_webmercator

WFS_URL = "https://geoegl.msp.gouv.qc.ca/ws/mffpecofor.fcgi"
BIO_LAYER = "ms:bio_arbv_tot"
CARB_LAYER = "ms:carb_arbv_tot"
BIO_FIELD = "b_arbv_tot"
CARB_FIELD = "c_arbv_tot"

AOI_BBOX = [-79.61448860036347, 48.32451136817797, -79.47908860036347, 48.414511368177976]
WGS84 = "EPSG:4326"
AREA_CRS = "EPSG:32198"  # Quebec Lambert, metres.

NS = {
    "wfs": "http://www.opengis.net/wfs/2.0",
    "gml": "http://www.opengis.net/gml/3.2",
    "ms": "http://mapserver.gis.umn.edu/mapserver",
}


def fetch_wfs_gml(layer: str) -> bytes:
    to_area = Transformer.from_crs(WGS84, AREA_CRS, always_xy=True).transform
    bbox_area = transform(to_area, box(*AOI_BBOX)).bounds
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typenames": layer,
        "outputFormat": "application/gml+xml; version=3.2",
        "srsName": WGS84,
        # Query in Quebec Lambert to avoid EPSG:4326 axis-order ambiguity.
        # Output geometries are requested in EPSG:4326 and served as lat lon.
        "bbox": f"{','.join(str(v) for v in bbox_area)},EPSG:32198",
    }
    response = requests.get(WFS_URL, params=params, timeout=60)
    response.raise_for_status()
    if b"ExceptionReport" in response.content:
        raise RuntimeError(response.text[:1000])
    return response.content


def parse_pos_list(pos_list_text: str) -> list:
    values = [float(v) for v in pos_list_text.split()]
    coords = []
    for i in range(0, len(values), 2):
        first, second = values[i], values[i + 1]
        # WFS returns EPSG:4326 as lat lon; convert to lon lat for shapely.
        if abs(first) <= 60 and abs(second) >= 60:
            coords.append((second, first))
        else:
            coords.append((first, second))
    return coords


def polygon_from_gml(poly_el: ET.Element) -> Polygon:
    exterior_el = poly_el.find(".//gml:exterior//gml:posList", NS)
    if exterior_el is None or not exterior_el.text:
        raise ValueError("Polygon has no exterior posList")
    exterior = parse_pos_list(exterior_el.text)
    interiors = []
    for interior_el in poly_el.findall(".//gml:interior//gml:posList", NS):
        if interior_el.text:
            interiors.append(parse_pos_list(interior_el.text))
    polygon = Polygon(exterior, interiors)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def geometry_from_feature(feature: ET.Element):
    polygons = [polygon_from_gml(poly_el) for poly_el in feature.findall(".//gml:Polygon", NS)]
    if not polygons:
        return None
    if len(polygons) == 1:
        return polygons[0]
    return MultiPolygon(polygons)


def parse_layer(gml_content: bytes, layer_name: str, value_field: str) -> dict:
    root = ET.fromstring(gml_content)
    local_name = layer_name.split(":")[-1]
    records = {}
    for feature in root.findall(f".//ms:{local_name}", NS):
        geocode_el = feature.find("ms:geocode", NS)
        value_el = feature.find(f"ms:{value_field}", NS)
        geom = geometry_from_feature(feature)
        if geocode_el is None or value_el is None or geom is None:
            continue
        value_text = value_el.text
        if value_text in (None, ""):
            continue
        records[geocode_el.text] = {"value": float(value_text), "geometry": geom}
    return records


def main():
    bio_records = parse_layer(fetch_wfs_gml(BIO_LAYER), BIO_LAYER, BIO_FIELD)
    carb_records = parse_layer(fetch_wfs_gml(CARB_LAYER), CARB_LAYER, CARB_FIELD)

    aoi_lonlat = box(*AOI_BBOX)
    to_area = Transformer.from_crs(WGS84, AREA_CRS, always_xy=True).transform
    aoi_area_geom = transform(to_area, aoi_lonlat)
    aoi_area_ha = aoi_area_geom.area / 10000

    rows = []
    for geocode, bio in bio_records.items():
        if geocode not in carb_records:
            continue
        intersection = bio["geometry"].intersection(aoi_lonlat)
        if intersection.is_empty:
            continue
        intersection_area_ha = transform(to_area, intersection).area / 10000
        if intersection_area_ha <= 0:
            continue
        rows.append({
            "geocode": geocode,
            "area_ha": intersection_area_ha,
            "biomass_Mg_ha": bio["value"],
            "carbon_MgC_ha": carb_records[geocode]["value"],
        })

    coverage_ha = sum(row["area_ha"] for row in rows)
    weighted_biomass = sum(row["biomass_Mg_ha"] * row["area_ha"] for row in rows) / coverage_ha if coverage_ha else 0
    weighted_carbon = sum(row["carbon_MgC_ha"] * row["area_ha"] for row in rows) / coverage_ha if coverage_ha else 0

    print("=== MRNF Quebec AGB/carbon cross-check, Abitibi analysis zone ===\n")
    print(f"WFS endpoint: {WFS_URL}")
    print(f"Biomass layer/field: {BIO_LAYER} / {BIO_FIELD}")
    print(f"Carbon layer/field: {CARB_LAYER} / {CARB_FIELD}")
    print("Pool: aboveground live-tree biomass/carbon; all species and all diameter classes; "
          "stump, stem, branches, foliage, wood, bark, leaves/needles included; roots excluded.")
    print("Spatial unit: peuplements ecoforestiers / stand polygons from MRNF forest compilations.")
    print("Weighting: area(intersection stand polygon ∩ AOI).")
    print("Component fields: the WFS layers used here expose total biomass and total carbon only; "
          "separate wood/bark/branches/foliage fields were not exposed by these WFS feature schemas.\n")

    print(f"AOI total area: {aoi_area_ha:,.1f} ha")
    print(f"MRNF valid coverage area: {coverage_ha:,.1f} ha")
    print(f"MRNF coverage of AOI: {100 * coverage_ha / aoi_area_ha if aoi_area_ha else 0:.1f}%")
    print(f"Intersecting peuplements with biomass+carbon values: {len(rows)}")
    print(f"Area-weighted aboveground live-tree biomass: {weighted_biomass:.2f} Mg biomass/ha")
    print(f"Area-weighted aboveground live-tree carbon: {weighted_carbon:.2f} Mg C/ha")
    print(f"Implied carbon fraction: {weighted_carbon / weighted_biomass if weighted_biomass else 0:.3f}")


if __name__ == "__main__":
    main()
