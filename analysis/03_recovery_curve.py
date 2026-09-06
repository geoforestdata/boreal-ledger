"""
03_recovery_curve.py

Post-harvest spectral recovery curve for the Abitibi analysis zone, built
via chronosequence (space-for-time substitution): a single recent-year
NBR composite (2024) grouped by years-since-disturbance (Hansen lossyear),
covering ages 0-24. This captures SPECTRAL recovery (canopy closure), not
full structural/biomass recovery to a mature stand (60-100+ years).

Age classes with fewer than MIN_PIXELS are dropped from the plotted curve
-- an earlier version without this filter showed spurious zigzagging
driven by classes with as few as 5 pixels. The area-by-age bar chart (all
classes, unfiltered) is kept as a separate result: it reflects the real
harvest-year footprint of the landscape, not sampling noise.

Run: python 03_recovery_curve.py
"""

import sys
sys.path.append("../src")

import ee
import pandas as pd
import matplotlib.pyplot as plt
from zones import get_abitibi_analysis_zone
from recovery_utils import get_years_since_disturbance, get_sentinel2_nbr_composite

PROJECT = "your-gee-project-id"  # replace with your own GEE project
REFERENCE_YEAR = 2024
MAX_AGE = 24
MIN_PIXELS = 100


def main():
    ee.Initialize(project=PROJECT)
    aoi = get_abitibi_analysis_zone()

    age_img = get_years_since_disturbance(aoi, reference_year=REFERENCE_YEAR, max_age=MAX_AGE)
    nbr_img = get_sentinel2_nbr_composite(REFERENCE_YEAR, aoi)
    combined = nbr_img.addBands(age_img)

    combined_reducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.count(), sharedInputs=True)
    grouped = combined.reduceRegion(
        reducer=combined_reducer.group(groupField=1, groupName="age"),
        geometry=aoi, scale=30, maxPixels=1e13, bestEffort=True,
    ).get("groups").getInfo()

    curve_df = pd.DataFrame(grouped).rename(columns={"mean": "NBR"}).sort_values("age").reset_index(drop=True)
    curve_df["area_ha"] = curve_df["count"] * (30 * 30) / 10000

    reliable_df = curve_df[curve_df["count"] >= MIN_PIXELS].copy()
    dropped_ages = sorted(curve_df.loc[curve_df["count"] < MIN_PIXELS, "age"].astype(int).tolist())

    print("=== Post-harvest recovery curve, Abitibi analysis zone ===\n")
    print(f"Age classes with sufficient sample (>= {MIN_PIXELS} px): "
          f"{len(reliable_df)} of {len(curve_df)}")
    print(f"Dropped (insufficient sample): {dropped_ages}\n")
    print(reliable_df[["age", "NBR", "count", "area_ha"]].to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(reliable_df["age"], reliable_df["NBR"], marker="o", color="darkgreen")
    ax1.set_xlabel("Years since harvest")
    ax1.set_ylabel("NBR")
    ax1.set_title(f"Spectral recovery (age classes with >= {MIN_PIXELS} px only)")
    ax1.text(0.02, 0.02, "Spectral recovery, not full structural/biomass recovery",
              transform=ax1.transAxes, fontsize=8, style="italic", color="gray")

    colors = ["lightgray" if c < MIN_PIXELS else "steelblue" for c in curve_df["count"]]
    ax2.bar(curve_df["age"], curve_df["area_ha"], color=colors)
    ax2.set_xlabel("Years since harvest")
    ax2.set_ylabel("Area (ha)")
    ax2.set_title("Harvested area by year (gray = insufficient sample)")

    plt.tight_layout()
    plt.savefig("../figures/recovery_curve.png", dpi=200)
    print("\nFigure saved to ../figures/recovery_curve.png")


if __name__ == "__main__":
    main()

