"""
embedding_utils.py

Uses Google's AlphaEarth Foundations Satellite Embedding dataset (64-band,
10 m, annual since 2017) for two things:

  1. Unsupervised forest-mask identification per zone (k-means clustering,
     cross-referenced with Hansen tree cover to label the forest cluster).
  2. A structural "signature" comparison between zones via cosine
     similarity of their mean embeddings — computed only over forest
     pixels (see the notebook), since the raw bounding-box mean is
     dominated by regional climate/geography signal rather than forest
     structure.

GEE asset: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
"""

import numpy as np
import ee

EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"


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

    return forest_mask, mean_treecover_per_cluster, forest_cluster_id


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

