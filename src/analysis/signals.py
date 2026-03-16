"""
Extract detailed TF-IDF signals per cluster for the web dashboard.

Each signal includes:
  - term:  discriminative word/bigram
  - raw:   total raw count across cluster reviews
  - tfidf: TF-IDF weight (higher = more distinctive to this cluster)
  - pct:   % of cluster reviews containing the term
"""

from collections import defaultdict

_NOISE = {
    "app", "apps", "use", "used", "using", "just", "like",
    "review", "helpful", "people", "person", "don",
    "emergent", "march", "2026", "ve", "doesn",
}


def extract_cluster_signals(
    reviews: list[dict],
    top_n: int = 6,
) -> dict[int, list[dict]]:
    """
    Compute discriminative TF-IDF vocabulary per cluster with detailed stats.

    Args:
        reviews: Review dicts with ``cluster_id`` and ``text`` keys.
        top_n:   Maximum signals per cluster.

    Returns:
        Dict mapping cluster_id → [{term, raw, tfidf, pct}, …].
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_texts: dict[int, list[str]] = defaultdict(list)
    for r in reviews:
        cid = r.get("cluster_id", -1)
        if cid is not None and cid != -1:
            cluster_texts[cid].append(r["text"])

    cluster_ids = sorted(cluster_texts)
    documents = [" ".join(cluster_texts[cid]) for cid in cluster_ids]

    if len(documents) < 2:
        return {cid: [] for cid in cluster_ids}

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=200,
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    result: dict[int, list[dict]] = {}

    for idx, cid in enumerate(cluster_ids):
        row = tfidf_matrix[idx].toarray()[0]
        top_indices = row.argsort()[::-1]

        texts = cluster_texts[cid]
        signals: list[dict] = []

        for j in top_indices:
            if row[j] <= 0 or len(signals) >= top_n:
                break
            term = feature_names[j]
            if term in _NOISE:
                continue

            raw = sum(t.lower().count(term) for t in texts)
            docs_with = sum(1 for t in texts if term in t.lower())
            pct = round(docs_with / len(texts) * 100, 1) if texts else 0

            signals.append({
                "term": term,
                "raw": raw,
                "tfidf": round(float(row[j]), 2),
                "pct": pct,
            })

        result[cid] = signals

    return result
