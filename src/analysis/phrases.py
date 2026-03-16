"""
Detect repeated phrases (bigrams / trigrams) across reviews within each cluster.

A phrase is "repeated" if it appears in 2+ independent reviews in the
same cluster — a strong signal that multiple users share the same pain.
"""

import re
from collections import Counter, defaultdict

_STOP_WORDS = {
    "the", "a", "an", "is", "it", "its", "and", "or", "but", "in", "on",
    "at", "to", "for", "of", "with", "by", "from", "this", "that", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "i", "me", "my", "we", "you", "he", "she", "they", "them",
    "not", "no", "so", "if", "just", "very", "too", "also", "don", "t",
}


def extract_cluster_phrases(
    reviews: list[dict],
    min_count: int = 2,
    top_n: int = 8,
) -> dict[int, dict[str, int]]:
    """
    For each cluster, find bigrams/trigrams appearing in 2+ independent reviews.

    Args:
        reviews:   Review dicts with ``cluster_id`` and ``text`` keys.
        min_count: Minimum review appearances to qualify as "repeated".
        top_n:     Maximum phrases to return per cluster.

    Returns:
        Dict mapping cluster_id → {phrase: review_count}.
    """
    clusters: dict[int, list[str]] = defaultdict(list)
    for r in reviews:
        cid = r.get("cluster_id", -1)
        if cid is not None and cid != -1:
            clusters[cid].append(r["text"].lower())

    result: dict[int, dict[str, int]] = {}

    for cid, texts in clusters.items():
        phrase_review_count: Counter = Counter()

        for text in texts:
            words = _tokenize(text)
            seen: set[str] = set()
            for n in (2, 3):
                for i in range(len(words) - n + 1):
                    gram = " ".join(words[i : i + n])
                    if gram not in seen and not _is_stopword_only(gram):
                        seen.add(gram)
                        phrase_review_count[gram] += 1

        repeated = {p: c for p, c in phrase_review_count.items() if c >= min_count}
        result[cid] = dict(sorted(repeated.items(), key=lambda x: -x[1])[:top_n])

    return result


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 1]


def _is_stopword_only(phrase: str) -> bool:
    return all(w in _STOP_WORDS for w in phrase.split())
