"""
Dynamic 5-level taxonomy generation using LLM.

Instead of hardcoded taxonomy maps, this module:
  1. Generates L1 vocabulary from actual cluster data via Groq LLM
  2. Assigns each cluster to the best L1 category
  3. Generates L2/L3/theme/subtheme for each cluster via LLM
  4. Loads previous vocabulary from taxonomy_history.json for stability

Falls back to keyword-based heuristics if the LLM is unavailable.
"""

import json
import os
from pathlib import Path

TAXONOMY_HISTORY_FILE = "taxonomy_history.json"


def load_taxonomy_history(path=TAXONOMY_HISTORY_FILE):
    """Load previous taxonomy snapshots from disk."""
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_taxonomy_history(history, path=TAXONOMY_HISTORY_FILE):
    """Save taxonomy snapshot to disk for future runs."""
    try:
        Path(path).write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"  Warning: could not save taxonomy history: {e}")


def generate_l1_vocabulary(cluster_summaries, previous_vocab=None):
    """
    Use LLM to propose 3-6 broad L1 labels based on cluster data.
    Optionally evolves a previous vocabulary for stability.
    """
    cluster_lines = []
    for c in cluster_summaries:
        kw = c.get("top_keywords", [])[:5]
        cluster_lines.append(
            f"- {c.get('cluster_label', '?')}: {c.get('review_count', 0)} reviews, "
            f"sentiment {c.get('avg_sentiment', 0):.2f}, "
            f"keywords: {', '.join(kw) if kw else 'none'}"
        )

    evolution_block = ""
    if previous_vocab:
        vocab_list = "\n".join(f"  - {v}" for v in previous_vocab)
        evolution_block = f"""
PREVIOUS VOCABULARY (from last run - evolve, don't discard):
{vocab_list}

Rules for evolution:
- Keep labels that still have strong supporting clusters.
- Add new labels only if a cluster clearly doesn't fit any existing label.
- Remove labels only if no cluster maps to them at all.
"""

    cluster_block = "\n".join(cluster_lines)
    prompt = f"""You are a senior product analyst building a taxonomy for user reviews.

CLUSTER SUMMARIES:
{cluster_block}
{evolution_block}
Generate 3-6 broad L1 area labels that best categorise ALL the clusters above.
These labels should:
- Be 2-4 words each
- Reflect the actual product areas visible in the reviews
- NOT be generic (avoid "App Issues", "General Problems", "User Feedback")

Return ONLY valid JSON - a list of label strings:
["Label One", "Label Two", "Label Three"]"""

    try:
        from .llm import call_groq
        raw = call_groq(prompt, max_tokens=300)
        vocab = json.loads(raw)
        if isinstance(vocab, list) and len(vocab) >= 2:
            print(f"  LLM generated L1 vocabulary: {vocab}")
            return vocab
    except Exception as e:
        print(f"  LLM L1 vocab failed ({e}), using keyword fallback")

    return _fallback_vocab_from_keywords(cluster_summaries)


def _fallback_vocab_from_keywords(cluster_summaries):
    """Rule-based L1 vocabulary when LLM is unavailable."""
    vocab = set()
    for c in cluster_summaries:
        label = c.get("cluster_label", "").lower()
        sent = c.get("avg_sentiment", 0)
        if any(w in label for w in ("billing", "credit", "money", "price", "cost")):
            vocab.add("Billing & Pricing")
        elif any(w in label for w in ("bug", "crash", "deploy", "blank", "error")):
            vocab.add("Technical Reliability")
        elif any(w in label for w in ("ai", "agent", "memory", "chat", "answer")):
            vocab.add("AI & Agent Quality")
        elif any(w in label for w in ("free", "onboard", "tier", "mislead")):
            vocab.add("Onboarding & Conversion")
        elif any(w in label for w in ("login", "network", "auth")):
            vocab.add("Access & Connectivity")
        elif sent > 0.2:
            vocab.add("User Satisfaction")
        else:
            vocab.add("Product Experience")
    return list(vocab) if vocab else ["Product Experience", "Technical Issues", "User Satisfaction"]


def assign_l1_globally(cluster_summaries, previous_vocab=None):
    """
    Generate L1 vocabulary, then assign each cluster to the best L1 label.
    Returns dict mapping cluster_id -> L1 label.
    """
    vocab = generate_l1_vocabulary(cluster_summaries, previous_vocab)

    cluster_lines = []
    for c in cluster_summaries:
        kw = c.get("top_keywords", [])[:5]
        cluster_lines.append(
            f"- ID={c.get('cluster_id', '?')}, label=\"{c.get('cluster_label', '?')}\", "
            f"keywords: {', '.join(kw) if kw else 'none'}"
        )
    cluster_block = "\n".join(cluster_lines)
    vocab_str = ", ".join(f'"{v}"' for v in vocab)

    prompt = f"""Assign each cluster to exactly one L1 category.

L1 CATEGORIES: [{vocab_str}]

CLUSTERS:
{cluster_block}

Return ONLY valid JSON - a dict mapping cluster ID to L1 label:
{{"0": "Category A", "1": "Category B"}}"""

    try:
        from .llm import call_groq
        raw = call_groq(prompt, max_tokens=400)
        mapping = json.loads(raw)
        if isinstance(mapping, dict):
            result = {}
            for cid_str, l1 in mapping.items():
                try:
                    result[int(cid_str)] = l1
                except (ValueError, TypeError):
                    pass
            if result:
                print(f"  LLM assigned L1 labels for {len(result)} clusters")
                return result, vocab
    except Exception as e:
        print(f"  LLM L1 assignment failed ({e}), using fallback")

    # Fallback: assign based on keyword matching
    result = {}
    for c in cluster_summaries:
        cid = c.get("cluster_id", -1)
        label = c.get("cluster_label", "").lower()
        assigned = vocab[0] if vocab else "Product Experience"
        for v in vocab:
            v_words = {w.lower() for w in v.split() if len(w) > 3}
            if any(w in label for w in v_words):
                assigned = v
                break
        result[cid] = assigned
    return result, vocab


def generate_cluster_taxonomy(cluster_label, keywords, avg_sentiment, review_count, l1_label, used_themes=None):
    """
    Generate L2/L3/theme/subtheme for a single cluster via LLM.
    Respects used_themes to ensure unique theme names across clusters.
    """
    kw_str = ", ".join(keywords[:6]) if keywords else "none"

    taken_block = ""
    if used_themes:
        taken_list = "\n".join(f"  - {t}" for t in used_themes)
        taken_block = f"""
CRITICAL RULE - UNIQUE NAMES:
The "theme" field MUST be unique. These theme names are already taken:
{taken_list}
Never use these or anything similar. Find the SPECIFIC sub-problem."""

    prompt = f"""Generate a taxonomy for this review cluster.

Cluster: "{cluster_label}"
L1 Category: "{l1_label}"
Keywords: {kw_str}
Avg sentiment: {avg_sentiment:.2f}
Review count: {review_count}
{taken_block}

Return ONLY valid JSON:
{{"l2": "Sub-category (2-3 words)", "l3": "Specific area (2-3 words)", "theme": "Named pattern (2-4 words)", "subtheme": "One sentence detail"}}"""

    try:
        from .llm import call_groq
        raw = call_groq(prompt, max_tokens=300)
        result = json.loads(raw)
        if isinstance(result, dict) and "l2" in result:
            result["l1"] = l1_label
            return result
    except Exception as e:
        print(f"  LLM taxonomy for '{cluster_label}' failed ({e}), using fallback")

    return _fallback_taxonomy(cluster_label, avg_sentiment, l1_label)


def _fallback_taxonomy(cluster_label, avg_sentiment, l1_label):
    """Keyword-based fallback taxonomy when LLM is unavailable."""
    label_lower = cluster_label.lower()

    if any(w in label_lower for w in ("billing", "credit", "money")):
        return {"l1": l1_label, "l2": "Monetisation", "l3": "Credit System",
                "theme": cluster_label, "subtheme": "Billing and credit related issues"}
    if any(w in label_lower for w in ("bug", "crash", "deploy", "blank")):
        return {"l1": l1_label, "l2": "Reliability", "l3": "App Stability",
                "theme": cluster_label, "subtheme": "Technical failures and crashes"}
    if any(w in label_lower for w in ("ai", "agent", "memory", "forgetful")):
        return {"l1": l1_label, "l2": "AI Agent", "l3": "Response Quality",
                "theme": cluster_label, "subtheme": "AI quality and memory issues"}
    if any(w in label_lower for w in ("free", "tier", "onboard")):
        return {"l1": l1_label, "l2": "Onboarding", "l3": "Free Tier",
                "theme": cluster_label, "subtheme": "Free tier and onboarding friction"}
    if avg_sentiment > 0.2:
        return {"l1": l1_label, "l2": "Satisfaction", "l3": "Positive Feedback",
                "theme": cluster_label, "subtheme": "Positive user experience"}

    return {"l1": l1_label, "l2": "General", "l3": cluster_label,
            "theme": cluster_label, "subtheme": "Auto-detected cluster"}


def assign_taxonomy(cluster_label, avg_sentiment=0.0):
    """
    Backward-compatible function for single-cluster taxonomy assignment.
    Used when LLM-based full taxonomy isn't available.
    """
    return _fallback_taxonomy(cluster_label, avg_sentiment, "Product Experience")
