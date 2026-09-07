# Boreal Ledger

Applied remote sensing for one managed boreal landscape in Abitibi-Temiscamingue, Quebec.

The repository asks a forestry question, not a generic mapping question: what can open geospatial products say about dated canopy loss, disturbance attribution, probable post-harvest spectral recovery, forest structure, and carbon uncertainty in a ~10,000 ha managed boreal AOI?

## Study Area

The analysis uses a single verified rectangle in Abitibi-Temiscamingue. It is an analysis AOI, not an administrative boundary. The AOI was selected because Hansen `treecover2000 > 50` covers most of the box and because a broader lake-dominated regional extent would dilute the forest signal.

```python
ABITIBI_10K_CENTER_WEBMERCATOR = (-8855108, 6168550)
ABITIBI_10K_HALF_WIDTH_LON = 0.0677
ABITIBI_10K_HALF_WIDTH_LAT = 0.0450
```

Source: [`src/zones.py`](src/zones.py)

## Workflow

1. **Dated canopy loss**: Hansen Global Forest Change identifies pixels with dated canopy loss since 2001.
2. **Disturbance attribution**: dated Hansen loss is intersected with NBAC fire perimeters by loss year; non-matching loss is treated as a non-fire residual and interpreted as probable harvest in this managed forest context.
3. **Recovery**: a 2024 Sentinel-2 NBR composite is grouped by years since Hansen loss, producing a probable post-harvest spectral recovery chronosequence.
4. **Forest structure**: ETH canopy height 2020 describes canopy structure over forest pixels. It is not converted to stand age.
5. **Carbon**: ESA CCI, SCANFI, MRNF Quebec, and SoilGrids are reported as separate products and pools.
6. **Uncertainty**: ESA and SCANFI are compared against the same MRNF productive-stand footprint to avoid comparing different surfaces.

## Validated Results

| Result | Value |
| --- | ---: |
| AOI size reported by Earth Engine | ~10,000 ha |
| Dated Hansen canopy loss since 2001 | 10.1% of AOI |
| Mean years since loss within dated fraction | 14.6 years |
| ETH canopy height over forest pixels | 17.7 m |
| Non-fire residual Hansen loss | 1,015.5 ha |
| Fire-associated Hansen loss | 0.0 ha |
| NBAC fire perimeters intersecting AOI, 1972-2023 | 0 |
| Recovery age classes retained after >=100 pixel filter | 13 of 24 |
| SoilGrids SOC stock, 0-30 cm | 59.8 Mg C/ha |

## Carbon Products

Aboveground biomass is product-dependent in this AOI. The estimates below should not be averaged or treated as a single definitive value.

| Source | Spatial scope | Biomass | Carbon |
| --- | --- | ---: | ---: |
| SCANFI v1.2 | Forested AOI | 68.44 Mg biomass/ha | 32.17 Mg C/ha |
| MRNF Quebec | Productive stands covered by MRNF, 2,489.6 ha (~25% AOI) | 102.97 Mg biomass/ha | 51.06 Mg C/ha |
| ESA CCI Method C | Forested AOI, Hansen forest-fraction weighted | 171.93 Mg biomass/ha | 80.81 Mg C/ha |
| SoilGrids | Whole AOI, SOC stock 0-30 cm | NA | 59.8 Mg C/ha |

`analysis/04_carbon.py` keeps ESA CCI Method C as the reported ESA procedure: ESA CCI AGB 2022 is converted from Mg biomass/ha to Mg C/ha with a 0.47 carbon fraction, Hansen forest fraction is explicitly aggregated to ESA support, and the mean is weighted by that forest fraction.

```python
forest_fraction = (
    forest_mask.unmask(0)
    .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
    .reproject(crs=esa_projection, scale=100)
    .rename("forest_fraction")
)
```

Source: [`analysis/04_carbon.py`](analysis/04_carbon.py)

## Common-Footprint Uncertainty

MRNF Quebec covers only the productive stands represented in its product inside this AOI: 2,489.6 ha and 431 peuplements. Because this is about 25% of the AOI, the strict comparison against ESA and SCANFI is made on the exact same MRNF productive-stand footprint.

| Comparison vs MRNF | Pearson | Spearman | Bias | RMSE |
| --- | ---: | ---: | ---: | ---: |
| ESA CCI | 0.643 | 0.556 | +67.89 Mg biomass/ha | 72.39 Mg biomass/ha |
| SCANFI | 0.573 | 0.554 | -36.32 Mg biomass/ha | 41.23 Mg biomass/ha |

SCANFI has lower absolute bias and RMSE than ESA relative to MRNF on this common productive-forest footprint. MRNF is still not ground truth; it is the most locally relevant provincial inventory-derived reference currently used in this repository.

## Code Walkthrough

Hansen loss year is treated as a dated canopy-loss signal only where the source has a positive `lossyear`.

```python
lossyear = ee.Image(HANSEN_ASSET).select("lossyear")
calendar_year = lossyear.add(2000).updateMask(lossyear.gt(0))
years_since = ee.Image.constant(current_year).subtract(calendar_year)
```

Source: [`src/stand_age_utils.py`](src/stand_age_utils.py)

Fire attribution is a year-matched overlay, not a nearest-fire or any-fire test.

```python
fire_year_img = fires.reduceToImage(
    properties=[year_field], reducer=ee.Reducer.first()
).rename("fire_year")

year_matches = loss_calendar_year.eq(fire_year_img).unmask(0)
```

Source: [`src/disturbance_utils.py`](src/disturbance_utils.py)

The recovery curve is spectral. Sentinel-2 NBR is calculated from near infrared and shortwave infrared bands, then grouped by Hansen years since disturbance.

```python
composite = collection.median().clip(aoi)
return composite.normalizedDifference(["B8", "B12"]).rename("NBR")
```

Source: [`src/recovery_utils.py`](src/recovery_utils.py)

MRNF, ESA, and SCANFI are compared by stand using area-weighted intersections.

```python
return (df[value_col] * df[weight_col]).sum() / df[weight_col].sum()
```

Source: [`analysis/10_agb_mrnf_common_footprint.py`](analysis/10_agb_mrnf_common_footprint.py)

## Repository Structure

| Path | Purpose |
| --- | --- |
| `analysis/01_stand_age.py` | Dated Hansen canopy loss and ETH canopy-height summary. |
| `analysis/02_disturbance_attribution.py` | NBAC year-matched fire attribution and non-fire residual loss. |
| `analysis/03_recovery_curve.py` | Sentinel-2 NBR chronosequence and recovery figure. |
| `analysis/04_carbon.py` | Consolidated carbon-product summary and CSV export. |
| `analysis/05_agb_sensitivity.py` | ESA CCI masking/support sensitivity documentation. |
| `analysis/06_agb_crosscheck_scanfi.py` | SCANFI cross-check against ESA Method C. |
| `analysis/07_agb_product_diagnostics.py` | Same-mask ESA/SCANFI diagnostics, distributions, height and forest-type checks. |
| `analysis/08_agb_crosscheck_mrnf.py` | MRNF Quebec productive-stand biomass/carbon extraction. |
| `analysis/09_agb_crosscheck_ntems.py` | Optional NTEMS access diagnostic; not required for the main workflow. |
| `analysis/10_agb_mrnf_common_footprint.py` | MRNF/ESA/SCANFI common-footprint and stand-level diagnostics. |
| `src/` | Shared AOI, disturbance, recovery, structure, soil, and carbon helpers. |
| `outputs/` | Validated CSV outputs used by the carbon comparison and website. |
| `docs/` | Static GitHub Pages research story. |

## Reproduce

Install the Python dependencies and authenticate Earth Engine:

```bash
pip install -r requirements.txt
earthengine authenticate
```

Set `PROJECT` at the top of each analysis script to your Earth Engine project, then run:

```bash
python analysis/01_stand_age.py
python analysis/02_disturbance_attribution.py
python analysis/03_recovery_curve.py
python analysis/04_carbon.py
python analysis/05_agb_sensitivity.py
python analysis/06_agb_crosscheck_scanfi.py
python analysis/07_agb_product_diagnostics.py
python analysis/08_agb_crosscheck_mrnf.py
python analysis/10_agb_mrnf_common_footprint.py
```

`analysis/09_agb_crosscheck_ntems.py` is optional. It documents official NTEMS access routes and can compute a local raster only if a suitable NTEMS raster is provided. It requires GDAL only when a local NTEMS raster is supplied.

## Limitations

- Hansen `lossyear` gives dated canopy loss since 2001. It does not date the undisturbed majority of the AOI.
- The non-fire residual is interpreted as probable harvest because no NBAC fire perimeters intersect the AOI, but it is still a residual category.
- NBR is spectral vegetation recovery, not biomass recovery, structural recovery, or stand age.
- ESA CCI, SCANFI, and MRNF have different spatial supports and product definitions. The common-footprint diagnostic reduces, but does not eliminate, product uncertainty.
- SoilGrids SOC is reported only for 0-30 cm.
- AGB + SOC is not reported as total ecosystem carbon or total forest carbon because belowground biomass, dead wood, litter or forest floor, and SOC below 30 cm are not fully represented.

## References

- Hansen Global Forest Change v1.13, `UMD/hansen/global_forest_change_2025_v1_13`.
- Canadian National Burned Area Composite (NBAC), Natural Resources Canada / Canadian Forest Service, `projects/sat-io/open-datasets/CA_FOREST/NBAC/nbac_1972_2023_20240530`.
- Sentinel-2 Surface Reflectance Harmonized, `COPERNICUS/S2_SR_HARMONIZED`.
- ESA CCI Biomass v6.0, Santoro & Cartus (2025), `projects/sat-io/open-datasets/ESA/ESA_CCI_AGB`.
- SoilGrids 2.0, Poggio et al. (2021), ISRIC global soil information.
- ETH Global Canopy Height 2020, Lang et al. (2022), `users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1`.
- SCANFI v1.2, Spatialized Canadian National Forest Inventory, Natural Resources Canada / Canadian Forest Service.
- MRNF Quebec, *Biomasse et carbone forestiers du Quebec meridional*, Direction des inventaires forestiers.

Published Canadian and Quebec boreal carbon literature is used here for ecological context and plausibility only. It is not used as validation by similarity; methodological support comes from explicit masking, spatial weighting, and cross-product diagnostics.
