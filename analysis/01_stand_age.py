"""
01_stand_age.py

Stand age structure for the Abitibi analysis zone (~10,000 ha, verified
~90% real forest cover). Two independent signals, reported separately
(NOT combined into a single fabricated "age" number):

  1. Years since last Hansen-detected harvest -- a direct, dated
     measurement, but only covers the ~10% of the zone disturbed since
     2001. The other ~90% is age-censored (undated), not necessarily old.
  2. Canopy height (ETH Global Canopy Height 2020) -- a descriptive
     statistic reported on its own. Converting height to age would
     require site-index-calibrated growth equations (Pothier & Savard,
     1998, the standard Quebec MRNF reference), which need a field-
     estimated site quality index (IQS) that this remote-sensing-only
     pipeline does not have. No age number is derived from height here.

Run: python 01_stand_age.py
"""

import sys
sys.path.append("../src")

import ee
from zones import get_abitibi_analysis_zone
from stand_age_utils import get_years_since_loss, get_canopy_height

PROJECT = "your-gee-project-id"  # replace with your own GEE project


def main():
    ee.Initialize(project=PROJECT)
    aoi = get_abitibi_analysis_zone()

    forest_mask = ee.Image("UMD/hansen/global_forest_change_2025_v1_13").select("treecover2000").gt(50)

    years_since = get_years_since_loss(aoi, current_year=2025)
    pct_dated = years_since.mask().reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=30, maxPixels=1e13, bestEffort=True,
    ).get("years_since_loss").getInfo()
    mean_age_dated = years_since.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=30, maxPixels=1e13, bestEffort=True,
    ).get("years_since_loss").getInfo()

    canopy = get_canopy_height(aoi).updateMask(forest_mask)
    height_stats = canopy.reduceRegion(
        reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.count(), sharedInputs=True),
        geometry=aoi, scale=10, maxPixels=1e13, bestEffort=True,
    ).getInfo()

    print("=== Stand age structure, Abitibi analysis zone (~10,000 ha) ===\n")
    print(f"Share of zone with a dated harvest since 2001 (Hansen): {pct_dated:.1%}")
    print(f"Mean years since harvest, within that dated share: {mean_age_dated:.1f} years")
    print(f"Mean canopy height, forest pixels (ETH 2020): "
          f"{height_stats['canopy_height_m_mean']:.1f} m "
          f"(n={height_stats['canopy_height_m_count']:,} pixels)")
    print("\nInterpretation: the zone is a mix of a smaller actively-managed "
          "segment (dated, younger) and a larger undated segment whose canopy "
          "height suggests mature forest -- see README for the Pothier-Savard "
          "caveat on converting height to a calibrated age.")


if __name__ == "__main__":
    main()

