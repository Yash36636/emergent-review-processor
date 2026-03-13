"""
HDBSCAN clustering + confidence scoring.

Uses scikit-learn's built-in HDBSCAN (>=1.3) — no C++ compiler required on Windows.

Why HDBSCAN over K-Means?
  - No need to specify number of clusters upfront
  - Labels noise as -1 instead of forcing every point into a cluster
  - Finds clusters of arbitrary shape and varying density
  - Provides per-point membership probabilities
"""

import numpy as np
from collections import defaultdict


def cluster_embeddings(
    reduced_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run HDBSCAN on UMAP-reduced embeddings.

    Args:
        reduced_embeddings: UMAP-reduced array (N, k).

    Returns:
        (labels, probabilities) — both np.ndarray of shape (N,).
        labels[i] == -1  →  noise point.
    """
    from sklearn.cluster import HDBSCAN

    print("Running HDBSCAN clustering...")
    # Target 5-8 meaningful clusters from ~175 reviews.
    # ~8% of dataset per cluster avoids fragmentation without being too coarse.
    min_size = max(8, len(reduced_embeddings) // 12)

    clusterer = HDBSCAN(
        min_cluster_size=min_size,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="leaf",   # "leaf" gives tighter, more even clusters
    )
    clusterer.fit(reduced_embeddings)

    labels        = clusterer.labels_
    probabilities = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = list(labels).count(-1)
    print(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

    return labels, probabilities


def compute_confidence(
    embeddings: np.ndarray,
    labels: np.ndarray,
    hdbscan_probs: np.ndarray,
) -> np.ndarray:
    """
    Blended confidence score per review.

        confidence = 0.6 * hdbscan_probability + 0.4 * (1 - normalised_centroid_distance)

    The dual-source score is more robust:
      - HDBSCAN probability  → density-based stability
      - Centroid proximity   → how "central" the review is to its theme

    Args:
        embeddings:    Reduced embeddings (N, k).
        labels:        Cluster labels (N,).
        hdbscan_probs: HDBSCAN membership probabilities (N,).

    Returns:
        np.ndarray of shape (N,) with confidence values in [0, 1].
    """
    n               = len(labels)
    confidence      = np.zeros(n)
    centroids       = _compute_centroids(embeddings, labels)

    for i, (label, hdb_p) in enumerate(zip(labels, hdbscan_probs)):
        if label == -1:
            continue

        centroid   = centroids[label]
        dist       = np.linalg.norm(embeddings[i] - centroid)
        max_dist   = max(
            np.linalg.norm(embeddings[j] - centroid)
            for j in range(n) if labels[j] == label
        ) or 1.0
        proximity  = 1.0 - dist / max_dist
        confidence[i] = round(0.6 * hdb_p + 0.4 * proximity, 4)

    return confidence


def find_representatives(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[int, int]:
    """
    For each cluster, return the index of the review closest to the centroid.
    This is the most "typical" review in the cluster — useful for PM summaries.

    Returns:
        Dict mapping cluster_id → review index.
    """
    centroids = _compute_centroids(embeddings, labels)
    reps: dict[int, int] = {}

    for cid, centroid in centroids.items():
        best_idx  = -1
        min_dist  = float("inf")
        for i in range(len(labels)):
            if labels[i] == cid:
                d = np.linalg.norm(embeddings[i] - centroid)
                if d < min_dist:
                    min_dist = d
                    best_idx = i
        reps[cid] = best_idx

    return reps


def _compute_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    centroids = {}
    for cid in set(labels):
        if cid == -1:
            continue
        mask = labels == cid
        centroids[cid] = embeddings[mask].mean(axis=0)
    return centroids
