"""
Auto-label clusters using TF-IDF top keywords + per-review keyword voting.

Strategy (three passes in order):
  1. Per-review keyword vote: each review votes for one theme based on which
     theme's keyword list has the most hits. Cluster label = theme with the most
     votes. Requires at least MIN_VOTES reviews to agree.
  2. TF-IDF keyword signal: do the cluster's distinctive keywords contain
     any theme-specific words? Single strong keywords are allowed here since
     TF-IDF already filtered out generic cross-cluster terms.
  3. Fallback heuristics: short review cluster? mostly positive? mostly negative?
"""

from collections import defaultdict

# ── Noise words ──────────────────────────────────────────────────────────────
# Excluded from TF-IDF keyword matching only (not from per-review voting).
_NOISE_WORDS = {
    "app", "apps", "use", "used", "using",
    "review", "helpful", "people", "person",
    "review helpful", "people review helpful", "person review helpful",
    "found helpful", "found review", "found this", "1 person",
    "march", "march 2026", "2026", "star", "stars", "rating",
    "emergent", "emergent ai", "emergent app",
    "make", "made", "just", "got", "get", "even", "really", "like",
}

# ── Per-review voting rules ───────────────────────────────────────────────────
# Each list contains keywords/phrases. For a review to vote for a theme,
# at least one phrase must appear in the review text (case-insensitive).
# Rules with more specific phrases should be listed FIRST.
_VOTE_RULES: list[tuple[str, list[str]]] = [
    ("Login & Network Issues", [
        "login", "sign in", "signin", "sign-in", "can't log", "cannot login",
        "network error", "network request", "network failed", "stuck on login",
        "stuck on logo", "not loading", "wi-fi", "mobile network",
        "waking up the agent", "woke up", "agent. please try again",
        "create an account", "forced to create",
    ]),
    ("App Bugs & Crashes", [
        "blank screen", "not deployed", "play store", "playstore",
        "google play", "deployment", "glitch", "crashed", "disappeared",
        "build failed", "never appeared", "opens to a blank",
        "app didn't got deployed", "showing its deployed",
        "not actually deployed", "app opens up to a blank",
    ]),
    ("Misleading Free Tier", [
        "not free", "says free", "free use", "not willing to buy",
        "buy credits", "credit exosted", "credits exhausted", "credit exhausted",
        "requires expo", "expo", "free but", "hidden cost", "free tier",
        "free use", "advertised free",
    ]),
    ("Billing & Credit Issues", [
        "lost credits", "costs spiralled", "eating $", "charged again",
        "double charged", "scam", "scammer", "rip off", "ripoff",
        "not worth the money", "waste of money", "eats credit",
        "losing credits", "lost money", "overcharged", "credits gone",
        "reloading app", "500 a day", "spending cap",
        "1month 249 subscription", "subscription but",
    ]),
    ("AI Quality Issues", [
        "forgetful", "no chat history", "slower than digging",
        "no accountability", "code flaws", "security flaw",
        "misunderstands", "garbage code", "soulless ai",
        "no memory", "no context", "inaccurate", "wrong answers",
        "gives irrelevant", "too many errors",
    ]),
    ("Support & Response Issues", [
        "support has disappeared", "no reply", "no response",
        "they do not reply", "unresponsive", "support is gone",
        "no one responds",
    ]),
    ("Positive Experience", [
        "enhanced my career", "great experience", "vibe coding tool",
        "vibe coding", "best coding app", "definitely the best",
        "love the app", "badhiya", "best app ever", "highly recommend",
        "god given", "eternally grow",
    ]),
]

MIN_VOTES = 2   # minimum reviews agreeing before we trust the label

# ── TF-IDF single-keyword fallback signals ────────────────────────────────────
# Used only after per-review voting fails. TF-IDF ensures these keywords are
# distinctive to THIS cluster (not generic across all clusters).
_KW_FALLBACK: list[tuple[str, list[str]]] = [
    ("Login & Network Issues",    ["login", "sign", "signin", "network", "stuck"]),
    ("App Bugs & Crashes",        ["blank screen", "playstore", "play store", "crash",
                                   "deployment", "glitch", "disappeared", "build failed"]),
    ("Misleading Free Tier",      ["free", "expo", "not free", "buy credits",
                                   "credit exhausted"]),
    ("AI Quality Issues",         ["ai", "forgetful", "soulless", "memory",
                                   "garbage code", "slow agent"]),
    ("Billing & Credit Issues",   ["money", "scam", "charged", "overcharged",
                                   "subscription cost", "rip off"]),
    ("Support & Response Issues", ["no reply", "no response", "unresponsive"]),
    ("Positive Experience",       ["career", "badhiya", "vibe coding", "wonderful",
                                   "awesome", "grateful", "enhanced"]),
]


def label_clusters(reviews: list[dict], labels) -> dict[int, dict]:
    """
    Assign a text label and top keywords to each cluster.

    Args:
        reviews: List of review dicts (must have "text" key).
        labels:  Array-like of cluster IDs (same length as reviews).

    Returns:
        Dict mapping cluster_id → {"keywords": [...], "label": str}.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_texts: dict[int, list[str]] = defaultdict(list)
    for review, label in zip(reviews, labels):
        cluster_texts[int(label)].append(review["text"])

    cluster_ids = [cid for cid in cluster_texts if cid != -1]
    documents   = [" ".join(cluster_texts[cid]) for cid in cluster_ids]

    cluster_info: dict[int, dict] = {}

    if len(documents) >= 2:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            max_features=300,
            min_df=1,
        )
        tfidf_matrix  = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        for i, cid in enumerate(cluster_ids):
            row         = tfidf_matrix[i].toarray()[0]
            top_indices = row.argsort()[::-1][:8]
            keywords    = [feature_names[j] for j in top_indices if row[j] > 0]
            label_str   = _infer_label(keywords, cluster_texts[cid])
            cluster_info[cid] = {"keywords": keywords[:5], "label": label_str}
    else:
        for cid in cluster_ids:
            cluster_info[cid] = {"keywords": [], "label": f"Cluster {cid}"}

    if -1 in cluster_texts:
        cluster_info[-1] = {"keywords": [], "label": "Noise / Unclassified"}

    return cluster_info


def _infer_label(keywords: list[str], texts: list[str]) -> str:
    """Three-pass label inference (majority vote → TF-IDF keyword → heuristic)."""

    # ── Pass 1: Per-review majority vote ─────────────────────────────────────
    votes: dict[str, int] = defaultdict(int)
    for text in texts:
        t = text.lower()
        best_label = None
        best_hits  = 0
        for label, phrases in _VOTE_RULES:
            hits = sum(1 for p in phrases if p in t)
            if hits > best_hits:
                best_hits  = hits
                best_label = label
        if best_label and best_hits > 0:
            votes[best_label] += 1

    if votes:
        top_votes = max(votes.values())
        total     = len(texts)
        # Only accept if there is a CLEAR winner (no tie) with enough votes
        top_candidates = [lbl for lbl, cnt in votes.items() if cnt == top_votes]
        if (len(top_candidates) == 1
                and top_votes >= MIN_VOTES
                and top_votes / total >= 0.08):
            return top_candidates[0]

    # ── Pass 2: TF-IDF keyword signal ────────────────────────────────────────
    # Use whole-word matching so "ai" doesn't match "hai" or "bait"
    import re as _re
    clean_kws = [k for k in keywords if k not in _NOISE_WORDS]
    kw_str    = " ".join(clean_kws).lower()

    for label, signals in _KW_FALLBACK:
        for sig in signals:
            pattern = r"(?<![a-z])" + _re.escape(sig) + r"(?![a-z])"
            if _re.search(pattern, kw_str):
                return label

    # ── Pass 3: Cluster composition heuristics ───────────────────────────────
    short_count    = sum(1 for t in texts if len(t.split()) <= 8)
    positive_words = {"awesome", "great", "love", "nice", "good", "excellent",
                      "best", "wonderful", "perfect", "amazing", "helpful"}
    negative_words = {"bad", "terrible", "worst", "awful", "useless", "sucks",
                      "waste", "garbage", "poor", "horrible", "disgusting"}

    pos_count = sum(1 for t in texts if any(w in t.lower() for w in positive_words))
    neg_count = sum(1 for t in texts if any(w in t.lower() for w in negative_words))

    if short_count / max(len(texts), 1) >= 0.4:
        return "General Positive / Short" if pos_count >= neg_count else "General Negative / Vague"

    # Last resort
    meaningful = [k for k in keywords if k not in _NOISE_WORDS]
    return f"Theme: {', '.join(meaningful[:2])}" if meaningful else "Miscellaneous"
