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

