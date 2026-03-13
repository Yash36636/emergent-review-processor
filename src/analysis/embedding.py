"""
Text embedding and dimensionality reduction.

Embedding  : sentence-transformers/all-MiniLM-L6-v2 (384-dim, CPU-friendly)
Reduction  : UMAP  → 5-dim for clustering, 2-dim for visualization
"""

import numpy as np


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Encode texts into L2-normalised 384-dim dense vectors.

    The model is downloaded on first run (~90 MB) and cached locally at:
    ~/.cache/torch/sentence_transformers/

    Args:
        texts: List of raw review strings.

    Returns:
        np.ndarray of shape (N, 384).
    """
    from sentence_transformers import SentenceTransformer

    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Embedding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    UMAP reduction: 384-dim → n_components-dim (for clustering)
                  + 384-dim → 2-dim             (for visualisation).

    Why reduce before HDBSCAN?
    In 384-dim space all points appear equally distant (curse of
    dimensionality), breaking density estimation. Reducing to 5-10 dims
    preserves local neighbourhood structure while making density meaningful.

    Args:
        embeddings:   L2-normalised sentence embeddings (N, 384).
        n_components: Target dims for clustering (default 5).

    Returns:
        (reduced_nd, coords_2d) — both np.ndarray.
    """
    import umap

    print(f"UMAP reduction: 384 -> {n_components}D for clustering...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)

    print("UMAP reduction: 384 -> 2D for visualisation...")
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=5,
        min_dist=0.15,
        metric="cosine",
        random_state=42,
    )
    coords_2d = reducer_2d.fit_transform(embeddings)

    return reduced, coords_2d
