"""
End-to-end ML review analysis pipeline.

Steps:
  1. Score each review (sentiment, severity, actionability)
  2. Embed reviews with sentence-transformers
  3. Reduce dimensions with UMAP (384D → 5D + 2D)
  4. Cluster with HDBSCAN
  5. Compute blended confidence scores
  6. Auto-label clusters via TF-IDF
  7. Find representative review per cluster
  8. Aggregate cluster statistics
  9. Write JSON output

Output JSON schema:
  {
    "meta":             { ... pipeline config ... },
    "cluster_summary":  [ { cluster stats }, ... ],  sorted by priority_score DESC
    "reviews":          [ { per-review data }, ... ]
  }
"""

import json
from collections import defaultdict

import numpy as np

from .scoring    import score_sentiment, score_severity, score_actionability
from .embedding  import embed_texts, reduce_dimensions
from .clustering import cluster_embeddings, compute_confidence, find_representatives
from .labelling  import label_clusters


def run_pipeline(reviews: list[dict], output_path: str) -> dict:
    """
    Run the full clustering & scoring pipeline.

    Args:
        reviews:     List of dicts with at minimum keys: id, name, date, text.
        output_path: File path for the JSON output.

    Returns:
        The complete output dict (also written to output_path).
    """
    _sep()
    print("EMERGENT REVIEW CLUSTERING PIPELINE")
    _sep()

    texts = [r["text"] for r in reviews]

    # Step 1: Score
    _step(1, "Scoring reviews")
    scored = []
    for r in reviews:
        scored.append({
            **r,
            "sentiment":          score_sentiment(r["text"]),
            "severity":           score_severity(r["text"]),
            "actionability":      score_actionability(r["text"]),
            "cluster_id":         None,
            "cluster_label":      None,
            "cluster_confidence": None,
            "is_noise":           None,
            "is_representative":  None,
            "umap_2d_x":          None,
            "umap_2d_y":          None,
        })
    print(f"  Scored {len(scored)} reviews.")

    # Step 2: Embed
    _step(2, "Embedding")
    embeddings = embed_texts(texts)
    print(f"  Shape: {embeddings.shape}")

    # Step 3: Reduce
    _step(3, "UMAP dimension reduction")
    reduced_5d, coords_2d = reduce_dimensions(embeddings, n_components=5)
    print(f"  Reduced shape: {reduced_5d.shape}")

    # Step 4: Cluster
    _step(4, "HDBSCAN clustering")
    labels, hdb_probs = cluster_embeddings(reduced_5d)

    # Step 5: Confidence
    _step(5, "Confidence scoring")
    confidence = compute_confidence(reduced_5d, labels, hdb_probs)

    # Step 6: Label
    _step(6, "Auto-labelling clusters")
    cluster_info = label_clusters(reviews, labels)
    for cid, info in sorted(cluster_info.items()):
        print(f"  Cluster {cid:2d}: {info['label']:<35s}  kw={info['keywords']}")

    # Step 7: Representatives
    reps = find_representatives(reduced_5d, labels)

    # Step 8: Attach to reviews
    _step(7, "Attaching cluster data")
    for i, (review, label, conf) in enumerate(zip(scored, labels, confidence)):
        cid = int(label)
        info = cluster_info.get(cid, {})
        review.update({
            "cluster_id":         cid,
            "cluster_label":      info.get("label", f"Cluster {cid}"),
            "cluster_confidence": float(conf),
            "is_noise":           cid == -1,
            "is_representative":  reps.get(cid) == i,
            "umap_2d_x":          round(float(coords_2d[i][0]), 6),
            "umap_2d_y":          round(float(coords_2d[i][1]), 6),
        })

    # Step 9: Aggregate
    _step(8, "Aggregating cluster statistics")
    cluster_summary = _aggregate(reviews, labels, cluster_info, scored)

    output = {
        "meta": {
            "total_reviews":   len(reviews),
            "n_clusters":      sum(1 for c in cluster_summary if not c["is_noise"]),
            "n_noise_points":  int(sum(1 for l in labels if l == -1)),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dims":  int(embeddings.shape[1]),
            "umap_dims":       5,
            "clustering_algo": "HDBSCAN (sklearn)",
            "hdbscan_params": {
                "min_cluster_size":          2,
                "min_samples":               1,
                "cluster_selection_epsilon": 0.3,
                "cluster_selection_method":  "eom",
            },
        },
        "cluster_summary": cluster_summary,
        "reviews":         scored,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nOutput written to: {output_path}")

    # Summary table
    _sep()
    print("CLUSTER PRIORITY RANKING")
    _sep()
    print(f"{'#':<3} {'Label':<35} {'N':>5}  {'Priority':>8}  {'Sentiment':>9}  {'HighSev':>7}")
    print("-" * 72)
    for rank, c in enumerate(cluster_summary, 1):
        if not c["is_noise"]:
            print(
                f"{rank:<3} {c['cluster_label']:<35} {c['review_count']:>5}  "
                f"{c['priority_score']:>8.3f}  {c['avg_sentiment']:>9.3f}  "
                f"{c['high_severity_count']:>7}"
            )

    return output


# ── Helpers ───────────────────────────────────────────────────────────────────

_SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def _aggregate(
    reviews: list[dict],
    labels,
    cluster_info: dict,
    scored: list[dict],
) -> list[dict]:
    clusters: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[int(label)].append(i)

    summary = []
    for cid, indices in clusters.items():
        cr   = [scored[i] for i in indices]
        info = cluster_info.get(cid, {})

        sent_scores = [r["sentiment"]["compound"] for r in cr]
        sev_scores  = [r["severity"]["score"] for r in cr]
        act_scores  = [r["actionability"]["score"] for r in cr]
        conf_scores = [r["cluster_confidence"] for r in cr]

        neg   = sum(1 for r in cr if r["sentiment"]["label"] == "Negative")
        pos   = sum(1 for r in cr if r["sentiment"]["label"] == "Positive")
        hi_sv = sum(1 for r in cr if r["severity"]["label"] in ("Critical", "High"))

        summary.append({
            "cluster_id":            cid,
            "cluster_label":         info.get("label", f"Cluster {cid}"),
            "is_noise":              cid == -1,
            "top_keywords":          info.get("keywords", []),
            "review_count":          len(indices),
            "review_ids":            [reviews[i]["id"] for i in indices],
            "avg_sentiment":         round(float(np.mean(sent_scores)), 4),
            "avg_severity_score":    round(float(np.mean(sev_scores)), 4),
            "avg_actionability":     round(float(np.mean(act_scores)), 4),
            "avg_confidence":        round(float(np.mean(conf_scores)), 4),
            "negative_review_count": neg,
            "positive_review_count": pos,
            "high_severity_count":   hi_sv,
            "priority_score":        round(
                0.5 * float(np.mean(sev_scores))
                + 0.3 * (neg / len(indices))
                + 0.2 * min(len(indices) / 10, 1.0),
                4,
            ),
        })

    summary.sort(key=lambda x: x["priority_score"], reverse=True)
    return summary


def _sep():
    print("\n" + "=" * 60)


def _step(n: int, name: str):
    print(f"\n-- Step {n}: {name} --")
