"""
Scoring functions for individual reviews.

Three independent scores are computed per review:
  - Sentiment  : VADER compound score + label (Positive / Neutral / Negative)
  - Severity   : Weighted keyword scan → 0–1 score + Critical/High/Medium/Low label
  - Actionability: Feature specificity + quantities + improvement phrases → 0–1 score
"""

import re


# ── Sentiment ─────────────────────────────────────────────────────────────────

_VADER_ANALYZER = None
_VADER_FAILED = False


def _get_vader_analyzer():
    """Lazy-load a single VADER instance to avoid MemoryError from repeated lexicon loads."""
    global _VADER_ANALYZER, _VADER_FAILED
    if _VADER_FAILED:
        return None
    if _VADER_ANALYZER is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _VADER_ANALYZER = SentimentIntensityAnalyzer()
        except (MemoryError, OSError) as e:
            _VADER_FAILED = True
            import warnings
            warnings.warn(f"VADER failed to load ({e}). Using rule-based fallback.")
            return None
    return _VADER_ANALYZER


def _fallback_sentiment(text: str) -> dict:
    """Lightweight rule-based sentiment when VADER fails."""
    tl = text.lower()
    pos = sum(1 for w in ("great","love","awesome","perfect","good","nice","amazing","excellent") if w in tl)
    neg = sum(1 for w in ("bad","terrible","awful","sucks","garbage","scam","worst","hate") if w in tl)
    c = 0.0
    if pos > neg:
        c = min(0.5 + (pos - neg) * 0.1, 1.0)
    elif neg > pos:
        c = max(-0.5 - (neg - pos) * 0.1, -1.0)
    return {
        "compound":       round(c, 4),
        "positive_ratio": round(max(0, c), 4) if c > 0 else 0,
        "negative_ratio": round(max(0, -c), 4) if c < 0 else 0,
        "neutral_ratio":  round(1 - abs(c), 4) if abs(c) < 1 else 0,
        "label":          "Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral"),
    }


def score_sentiment(text: str) -> dict:
    """
    VADER sentiment analysis with singleton analyzer and MemoryError fallback.
    """
    analyzer = _get_vader_analyzer()
    if analyzer is not None:
        try:
            scores = analyzer.polarity_scores(text)
            compound = scores["compound"]
            return {
                "compound":       round(compound, 4),
                "positive_ratio": round(scores["pos"], 4),
                "negative_ratio": round(scores["neg"], 4),
                "neutral_ratio":  round(scores["neu"], 4),
                "label":          "Positive" if compound >= 0.05 else ("Negative" if compound <= -0.05 else "Neutral"),
            }
        except MemoryError:
            pass
    return _fallback_sentiment(text)


# ── Severity ──────────────────────────────────────────────────────────────────

_SEVERITY_SIGNALS = {
    "critical": {
        "keywords": [
            "scam", "scammed", "fraud", "stole", "disappeared", "blank screen",
            "lost money", "lost credits", "lost all", "charged again", "$500",
            "not deployed", "never appeared", "all disappeared", "launch for themselves",
        ],
        "weight": 1.0,
    },
    "high": {
        "keywords": [
            "lost", "gone", "nothing", "doesn't work", "doesn't deploy",
            "irrelevant", "misunderstands", "not worth", "alternatives",
            "garbage", "sucks", "useless", "forgetful", "no chat history",
            "no accountability", "wasted", "recharge", "eating credits",
        ],
        "weight": 0.6,
    },
    "medium": {
        "keywords": [
            "too fast", "expensive", "spendy", "too much", "confusing",
            "improvement", "not free", "costs", "pricy", "slow", "glitches",
        ],
        "weight": 0.3,
    },
    "low": {
        "keywords": ["bad", "not good", "ok", "average", "boring"],
        "weight": 0.1,
    },
}


def score_severity(text: str) -> dict:
    """
    Weighted keyword scan across severity tiers.
    Score is capped at 1.0; label maps from score thresholds.
    """
    text_lower      = text.lower()
    raw_score       = 0.0
    matched_signals = []

    for tier, config in _SEVERITY_SIGNALS.items():
        for kw in config["keywords"]:
            if kw in text_lower:
                raw_score += config["weight"]
                matched_signals.append({"keyword": kw, "tier": tier, "weight": config["weight"]})

    normalized = min(round(raw_score, 4), 1.0)

    if normalized >= 0.8:
        label = "Critical"
    elif normalized >= 0.4:
        label = "High"
    elif normalized >= 0.15:
        label = "Medium"
    else:
        label = "Low"

    return {"score": normalized, "label": label, "matched_signals": matched_signals}


# ── Actionability ─────────────────────────────────────────────────────────────

_ACTIONABILITY_SIGNALS = {
    "specific_feature": [
        "chat history", "memory", "deploy", "playstore", "play store",
        "credits", "billing", "support", "response", "hinglish", "expo",
        "blank screen", "accuracy", "subscription", "contact number",
    ],
    "quantified": [
        r"\$\d+", r"\d+ credits", r"\d+ times", r"\d+ day", r"100 credits",
    ],
    "suggests_improvement": [
        "please improve", "should be", "needs to", "would be better",
        "wish it had", "jarurat hai", "improvement",
    ],
    "vague_noise": [
        r"^(ok|nice|bad|good|awesome|sucks?|vary bad|bakwaas)\.?$",
    ],
}


def score_actionability(text: str) -> dict:
    """
    Measures how actionable a review is for product/engineering teams.
    Penalises vague single-word reviews; rewards specifics and quantities.
    Returns 0–1 score.
    """
    text_lower   = text.lower().strip()
    score        = 0.0
    signals_found = []

    for pattern in _ACTIONABILITY_SIGNALS["vague_noise"]:
        if re.match(pattern, text_lower):
            return {"score": 0.0, "label": "Not actionable", "signals": ["vague_single_word"]}

    for kw in _ACTIONABILITY_SIGNALS["specific_feature"]:
        if kw in text_lower:
            score += 0.25
            signals_found.append(kw)

    for pattern in _ACTIONABILITY_SIGNALS["quantified"]:
        if re.search(pattern, text_lower):
            score += 0.3
            signals_found.append(f"quantified:{pattern}")

    for phrase in _ACTIONABILITY_SIGNALS["suggests_improvement"]:
        if phrase in text_lower:
            score += 0.2
            signals_found.append(phrase)

    word_count = len(text.split())
    if word_count > 30:
        score += 0.2
    elif word_count > 10:
        score += 0.1

    normalized = min(round(score, 4), 1.0)

    if normalized >= 0.6:
        label = "Highly actionable"
    elif normalized >= 0.3:
        label = "Actionable"
    elif normalized > 0.0:
        label = "Low actionability"
    else:
        label = "Not actionable"

    return {"score": normalized, "label": label, "signals": list(set(signals_found))}
