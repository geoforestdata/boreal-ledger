"""
02_disturbance_attribution.py

Splits Hansen-detected forest loss (2001-2023) in the Abitibi analysis
zone into fire-caused vs non-fire (harvest) disturbance, using the
Canadian National Burned Area Composite (NBAC). Also runs a direct
diagnostic (fires touching the zone in the full 1972-2023 NBAC record,
regardless of year match) so a "0 ha fire" result can be distinguished
from a data-matching bug rather than assumed to be real.

Run: python 02_disturbance_attribution.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ee
from zones import get_abitibi_analysis_zone
from disturbance_utils import get_disturbance_cause_map, NBAC_COLLECTION

PROJECT = "ibfra2026"  # replace with your own GEE project


def main():
    ee.Initialize(project=PROJECT)
    aoi = get_abitibi_analysis_zone()
    pixel_ha = ee.Image.pixelArea().divide(10000)

    cause_map = get_disturbance_cause_map(aoi, year_field="YEAR")
    cause_stats = cause_map.multiply(pixel_ha).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e13, bestEffort=True,
    ).getInfo()
    fire_ha = cause_stats.get("fire_loss", 0) or 0
    non_fire_ha = cause_stats.get("non_fire_loss", 0) or 0

    fires_nearby = ee.FeatureCollection(NBAC_COLLECTION).filterBounds(aoi)
    n_fires_any_year = fires_nearby.size().getInfo()

    print("=== Disturbance attribution, Abitibi analysis zone (~10,000 ha) ===\n")
    print(f"Non-fire residual loss (probable harvest in this managed forest context): {non_fire_ha:,.1f} ha")
    print(f"Fire: {fire_ha:,.1f} ha")
    print(f"\nDiagnostic -- NBAC fire perimeters touching this zone, any year "
          f"1972-2023 (regardless of matching the loss year): {n_fires_any_year}")
    if fire_ha == 0 and n_fires_any_year == 0:
        print("The 0 ha fire result is confirmed real (no fires recorded here "
              "in 51 years) -- not a year-matching artifact.")
    elif fire_ha == 0 and n_fires_any_year > 0:
        print("WARNING: fires exist near this zone but none matched a loss "
              "year -- check the NBAC year_field or loss-year alignment "
              "before trusting the 0 ha fire result.")


if __name__ == "__main__":
    main()
