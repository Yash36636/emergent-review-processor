"""
Text embedding and dimensionality reduction.

Embedding  : sentence-transformers/all-MiniLM-L6-v2 (384-dim, CPU-friendly)
             Fallback: TF-IDF + TruncatedSVD when sentence-transformers fails (MemoryError)
Reduction  : UMAP  → 5-dim for clustering, 2-dim for visualization
"""

import numpy as np


def _embed_with_tfidf_svd(texts: list[str], n_components: int = 128) -> np.ndarray:
    """
    Lightweight fallback: TF-IDF + TruncatedSVD (LSA).
    Uses only sklearn — no PyTorch/transformers. Safe on memory-constrained systems.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    print("  Using TF-IDF + SVD fallback (lightweight, no neural model)...")
    vec = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    tfidf = vec.fit_transform(texts)
    n_comp = max(1, min(n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    dense = svd.fit_transform(tfidf)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (dense / norms).astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Encode texts into L2-normalised dense vectors.
    Primary: sentence-transformers (384-dim). Fallback: TF-IDF+SVD (128-dim).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except (MemoryError, OSError, ImportError) as e:
        import warnings
        warnings.warn(f"sentence-transformers failed to load ({e}). Using TF-IDF+SVD fallback.")
        return _embed_with_tfidf_svd(texts)

    try:
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
    except (MemoryError, OSError) as e:
        import warnings
        warnings.warn(f"sentence-transformers failed ({e}). Using TF-IDF+SVD fallback.")
        return _embed_with_tfidf_svd(texts)


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
