"""
Map cluster labels to a 5-level adaptive taxonomy.

Levels
------
  L1 → Broad category   (e.g. "App Experience")
  L2 → Sub-category     (e.g. "Monetisation")
  L3 → Specific area    (e.g. "Credit System")
  Theme → Named pattern  (e.g. "Credit Drain")
  Subtheme → Detail      (e.g. "Unexpected charges & credit loss")
"""

_TAXONOMY_MAP: dict[str, dict[str, str]] = {
    "Billing & Credit Issues": {
        "l1": "App Experience", "l2": "Monetisation", "l3": "Credit System",
        "theme": "Credit Drain",
        "subtheme": "Unexpected charges & credit loss",
    },
    "AI Quality Issues": {
        "l1": "App Experience", "l2": "AI Agent", "l3": "Memory & Context",
        "theme": "Agent Forgetfulness",
        "subtheme": "No persistent chat history, irrelevant responses",
    },
    "App Bugs & Crashes": {
        "l1": "App Experience", "l2": "Reliability", "l3": "Deployment Pipeline",
        "theme": "Deploy Failures",
        "subtheme": "Blank screen after payment, Play Store sync broken",
    },
    "Misleading Free Tier": {
        "l1": "Acquisition", "l2": "Onboarding", "l3": "Free Tier Clarity",
        "theme": "Misleading CTA",
        "subtheme": "App advertised free but requires immediate credit purchase",
    },
    "Positive Experience": {
        "l1": "App Experience", "l2": "Satisfaction", "l3": "Overall Delight",
        "theme": "Vibe Coding",
        "subtheme": "Positive prompt engineering & vibe coding experience",
    },
    "Login & Network Issues": {
        "l1": "App Experience", "l2": "Reliability", "l3": "Authentication",
        "theme": "Login Failures",
        "subtheme": "Network errors and authentication blocks",
    },
    "Support & Response Issues": {
        "l1": "App Experience", "l2": "Support", "l3": "Response Time",
        "theme": "Silent Support",
        "subtheme": "No replies to user queries or reviews",
    },
    "General Negative / Vague": {
        "l1": "App Experience", "l2": "Dissatisfaction", "l3": "Unspecified Friction",
        "theme": "Vague Negative",
        "subtheme": "Low-signal complaints requiring in-app follow-up prompt",
    },
    "General Positive / Short": {
        "l1": "App Experience", "l2": "Satisfaction", "l3": "General Positive",
        "theme": "Quick Praise",
        "subtheme": "Short positive reviews with limited detail",
    },
}


def assign_taxonomy(cluster_label: str, avg_sentiment: float = 0.0) -> dict[str, str]:
    """
    Return ``{l1, l2, l3, theme, subtheme}`` for a cluster label.
    Falls back to generic categories based on sentiment polarity.
    """
    if cluster_label in _TAXONOMY_MAP:
        return dict(_TAXONOMY_MAP[cluster_label])

    label_lower = cluster_label.lower()
    for key, tax in _TAXONOMY_MAP.items():
        key_words = {w.lower() for w in key.split() if len(w) > 3}
        if any(w in label_lower for w in key_words):
            return dict(tax)

    if avg_sentiment > 0.05:
        return {
            "l1": "App Experience", "l2": "Satisfaction", "l3": "General Positive",
            "theme": "Positive Feedback",
            "subtheme": "General positive sentiment",
        }

    if any(w in label_lower for w in ("vague", "general", "short", "noise")):
        return dict(_TAXONOMY_MAP["General Negative / Vague"])

    return {
        "l1": "App Experience", "l2": "Uncategorised", "l3": cluster_label,
        "theme": cluster_label,
        "subtheme": "Auto-detected cluster",
    }
