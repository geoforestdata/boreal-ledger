"""
embedding_utils.py

Uses Google's AlphaEarth Foundations Satellite Embedding dataset (64-band,
10 m, annual since 2017) to compute a mean embedding "signature" per zone
and compare zones by cosine similarity. This does not estimate carbon —
it is a complementary check on how spectrally/structurally distinct the
four sites are (e.g. an even-aged plantation should look more uniform and
more different from a structurally complex native forest than two native
forests of the same type would look from each other).

GEE asset: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
"""

import numpy as np
import ee

EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"


def get_mean_embedding(year: int, aoi: ee.Geometry, scale: int = 10) -> np.ndarray:
    """
    Returns the mean 64-band embedding vector over the zone for the given
    year, as a numpy array of length 64.
    """
    image = (
        ee.ImageCollection(EMBEDDING_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(aoi)
        .mosaic()
    )
    band_names = image.bandNames().getInfo()
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()
    return np.array([stats.get(b, 0.0) or 0.0 for b in band_names])


def cluster_and_identify_forest(
    year: int,
    aoi: ee.Geometry,
    n_clusters: int = 4,
    treecover_threshold: int = 50,
    scale: int = 10,
    training_pixels: int = 3000,
    seed: int = 0,
) -> ee.Image:
    """
    Unsupervised k-means clustering of the AlphaEarth embedding image over
    the zone, then labels whichever cluster has the highest mean Hansen
    tree-cover (2000) as "forest" and returns a binary mask (1 = forest).

    This exists to strip roads, clearings, water, and secondary/non-forest
    cover out of a raw bounding-box average before computing carbon density
    — a bare rectangle otherwise mixes land covers and biases the mean.
    """
    embedding_img = (
        ee.ImageCollection(EMBEDDING_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(aoi)
        .mosaic()
        .clip(aoi)
    )

    training = embedding_img.sample(
        region=aoi, scale=scale, numPixels=training_pixels, seed=seed, geometries=False
    )
    clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(training)
    clustered = embedding_img.cluster(clusterer)

    treecover = ee.Image("UMD/hansen/global_forest_change_2025_v1_13").select("treecover2000")

    cluster_ids = list(range(n_clusters))
    mean_treecover_per_cluster = {}
    for cid in cluster_ids:
        cluster_mask = clustered.eq(cid)
        mean_tc = treecover.updateMask(cluster_mask).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=scale,
            maxPixels=1e13, bestEffort=True,
        ).get("treecover2000").getInfo()
        mean_treecover_per_cluster[cid] = mean_tc or 0

    forest_cluster_id = max(mean_treecover_per_cluster, key=mean_treecover_per_cluster.get)
    forest_mask = clustered.eq(forest_cluster_id).selfMask().rename("forest_mask")
    # cluster() output does not inherit a default projection; reduceResolution
    # (used downstream in get_coarse_forest_mask) requires one explicitly.
    forest_mask = forest_mask.setDefaultProjection(embedding_img.projection())

    return forest_mask, mean_treecover_per_cluster, forest_cluster_id


def get_coarse_forest_mask(
    forest_mask_10m: ee.Image,
    target_image: ee.Image,
    min_fraction: float = 0.7,
) -> ee.Image:
    """
    Aggregates a fine-resolution (10 m) binary forest mask up to the native
    grid of a coarser-resolution target image (e.g. GEDI L4B at 1 km) by
    computing the FRACTION of forest pixels within each coarse cell, then
    thresholding.

    This matters because naively combining a 10 m mask with a 1 km image in
    reduceRegion uses nearest-neighbor resampling for the mask, not an area
    -weighted fraction — a coarse cell that is 40% lake / 60% forest can be
    included or excluded almost arbitrarily depending on which single 10 m
    pixel happens to land at the resampling point. Thresholding on the true
    forest fraction instead fully excludes mixed cells (e.g. forest edges,
    lakes, coastline) rather than letting them in with a distorted value.
    """
    forest_binary = forest_mask_10m.unmask(0)
    target_proj = target_image.projection()

    forest_fraction = (
        forest_binary
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=65536)
        .reproject(crs=target_proj)
    )
    return forest_fraction.gte(min_fraction)


def cosine_similarity_matrix(embeddings: dict) -> tuple:
    """
    Given a dict of {zone_key: embedding_vector}, returns (labels, matrix)
    where matrix[i, j] is the cosine similarity between zone i and zone j
    (1.0 = identical direction, 0.0 = orthogonal/unrelated).
    """
    labels = list(embeddings.keys())
    vectors = np.array([embeddings[k] for k in labels])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.where(norms == 0, 1, norms)
    matrix = normalized @ normalized.T
    return labels, matrix

