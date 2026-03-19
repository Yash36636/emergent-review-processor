"""
Flask web server for the Emergent Review Intelligence Dashboard.

Routes
------
  GET  /                    → Dashboard HTML (data injected via Jinja2)
  GET  /api/data            → JSON API for cluster + review data
  POST /api/chat            → Proxy to Groq API for the Wisdom AI chatbot
  POST /api/generate-insight→ Generate a 7-section PM insight report via Groq
"""

import json
import os
import urllib.request
import urllib.error
from pathlib import Path

from flask import Flask, render_template, request, jsonify

_TEMPLATE_DIR = Path(__file__).parent / "templates"

_CLUSTER_COLORS = [
    "#f05454", "#9b70f5", "#f0a030", "#f07840",
    "#3fbe7a", "#4da8f5", "#f0c454", "#e056a0",
    "#56c8c8", "#a0a0ff", "#80c060", "#d07030",
]


def create_app(pipeline_data: dict) -> Flask:
    """
    Build the Flask application.

    ``pipeline_data`` is the dict returned by ``run_pipeline()`` — the same
    data that drives the Excel report also drives the web dashboard.
    """
    app = Flask(__name__, template_folder=str(_TEMPLATE_DIR))
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB — chat payload can be large (clusters + reviews)

    clusters, reviews = _transform_pipeline_data(pipeline_data)

    @app.route("/")
    def dashboard():
        return render_template(
            "dashboard.html",
            clusters_json=json.dumps(clusters, ensure_ascii=False),
            reviews_json=json.dumps(reviews, ensure_ascii=False),
        )

    @app.route("/api/data")
    def api_data():
        return jsonify({"clusters": clusters, "reviews": reviews})

    def _call_groq_direct(payload: dict) -> tuple[dict, int]:
        """Call Groq API directly. Returns (data, status_code)."""
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            return {"error": "GROQ_API_KEY not set in .env"}, 400
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read()), 200
        except urllib.error.HTTPError as e:
            try:
                err_body = json.loads(e.read().decode("utf-8", errors="replace"))
            except Exception:
                err_body = {"error": f"HTTP {e.code}"}
            return err_body, min(e.code, 502)
        except Exception as e:
            return {"error": str(e)}, 502

    def _call_ai_api(payload: dict) -> tuple[dict, int]:
        """Call Groq API via proxy (if running) or directly. Returns (data, status_code)."""
        proxy_url = os.environ.get("AI_PROXY_URL", "").strip()
        use_proxy = proxy_url and os.environ.get("USE_AI_PROXY", "").lower() in ("1", "true", "yes")

        if use_proxy:
            try:
                req = urllib.request.Request(
                    f"{proxy_url.rstrip('/')}/api/grok",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=90) as resp:
                    return json.loads(resp.read()), 200
            except urllib.error.HTTPError as e:
                try:
                    err_body = json.loads(e.read().decode("utf-8", errors="replace"))
                except Exception:
                    err_body = {"error": f"HTTP {e.code}"}
                return err_body, min(e.code, 502)
            except (OSError, urllib.error.URLError, TimeoutError) as e:
                # Proxy unreachable — fallback to direct Groq
                print(f"  [AI] Proxy unreachable ({e}), falling back to Groq direct")
                return _call_groq_direct(payload)
            except Exception as e:
                print(f"  [AI] Proxy error: {e}, falling back to Groq direct")
                return _call_groq_direct(payload)
        return _call_groq_direct(payload)

    @app.route("/api/chat", methods=["POST"])
    def chat_proxy():
        body = request.get_json(force=True)
        data, status = _call_ai_api(body)
        if "choices" not in data:
            data = {"choices": [{"message": {"content": data.get("error", "Unknown error")}}]}
        return jsonify(data), status

    @app.route("/api/generate-insight", methods=["POST"])
    def generate_insight():
        payload = request.get_json(force=True)
        cluster_id = payload.get("cluster_id")

        cluster = next((c for c in clusters if c["id"] == cluster_id), None)
        if not cluster:
            return jsonify({"error": f"Cluster {cluster_id} not found."}), 404

        cluster_reviews = [r for r in reviews if r["cid"] == cluster_id]
        total_reviews = len(reviews)
        prompt = _build_insight_prompt(cluster, cluster_reviews, total_reviews)

        ai_payload = {
            "model": "llama-3.1-8b-instant",  # higher free-tier limits than 70b
            "max_tokens": 2500,
            "messages": [{"role": "user", "content": prompt}],
        }
        data, status = _call_ai_api(ai_payload)
        if status >= 400:
            return jsonify({"error": data.get("error", data.get("message", str(data)))}), status
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return jsonify({"insight": text, "cluster_id": cluster_id})

    return app


def _build_insight_prompt(cluster: dict, cluster_reviews: list, total_reviews: int) -> str:
    signals_text = ", ".join(
        f'"{s["term"]}" (TF-IDF {s["tfidf"]:.2f}, in {s["pct"]:.0f}% of reviews)'
        for s in cluster.get("signals", [])[:6]
    )
    phrases_text = ", ".join(
        f'"{p}" ({c}x)' for p, c in list(cluster.get("phrases", {}).items())[:5]
    )

    sample_reviews = []
    reps = [r for r in cluster_reviews if r.get("rep")]
    others = [r for r in cluster_reviews if not r.get("rep")]
    for r in (reps + others)[:8]:
        sample_reviews.append(f'  - [{r["sent"]["l"]}, sev {r["sev"]["s"]}] "{r["text"][:300]}"')
    reviews_block = "\n".join(sample_reviews)

    pct_of_total = (cluster["count"] / total_reviews * 100) if total_reviews else 0

    return f"""You are a senior product intelligence analyst. Analyse this cluster of app reviews and produce a structured actionable insight report.

## Cluster Data
- **Name:** {cluster["label"]}
- **Review count:** {cluster["count"]} ({pct_of_total:.1f}% of all feedback)
- **Avg sentiment:** {cluster["sentiment"]} (scale: -1 to +1)
- **Avg severity:** {cluster["severity"]} / 10
- **Avg actionability:** {cluster["actionability"]}
- **Negative reviews:** {cluster["neg"]} · Positive: {cluster["pos"]}
- **High-severity reviews:** {cluster["hsev"]}
- **Taxonomy:** {cluster["l1"]} → {cluster["l2"]} → {cluster["l3"]} → {cluster["theme"]} → {cluster["subtheme"]}

## Discriminative Signals (TF-IDF)
{signals_text or "None extracted."}

## Repeated Phrases
{phrases_text or "None detected."}

## Sample Reviews
{reviews_block}

---

Respond in EXACTLY this JSON format (no markdown fences, just raw JSON):
{{
  "problem_statement": "2-3 sentence plain-language description of the broken user experience — not keywords, but the actual pain",
  "evidence": "Quantitative summary: review count, sentiment, severity, and 2-3 direct representative quotes from the reviews above",
  "root_cause_hypothesis": "Your best inference about WHY this exists in the product — a structural/design/technical cause, not just restating the symptom",
  "user_impact": "Concrete effects on trust, churn, conversion, app store rating — be specific and directional",
  "recommended_actions": ["Action 1: specific implementable fix", "Action 2: ...", "Action 3: ..."],
  "priority": "Critical|High|Medium|Low",
  "priority_reasoning": "1-2 sentences explaining the priority level based on severity, volume, and sentiment",
  "slack_summary": "A ready-to-paste Slack message (plain text, with emoji). Format:\\n🔴/🟡/🟢 Cluster Name — PRIORITY\\n\\nProblem in 2 lines.\\n\\nStats line.\\nRepeated phrase line.\\n\\nTop fix recommendation.\\n\\n— Generated by Emergent VoC Intelligence"
}}"""


# ── Pipeline → Dashboard transformer ────────────────────────────────────────

def _transform_pipeline_data(data: dict) -> tuple[list, list]:
    """Convert ``run_pipeline()`` output to the dashboard's CLUSTERS / REVIEWS."""
    cluster_summary = data.get("cluster_summary", [])
    pipeline_reviews = data.get("reviews", [])

    clusters = []
    for i, cs in enumerate(cluster_summary):
        if cs.get("is_noise"):
            continue
        taxonomy = cs.get("taxonomy", {})
        clusters.append({
            "id":            cs["cluster_id"],
            "label":         cs["cluster_label"],
            "color":         cs.get("color", _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]),
            "priority":      round(cs["priority_score"], 2),
            "count":         cs["review_count"],
            "sentiment":     round(cs["avg_sentiment"], 2),
            "severity":      round(cs["avg_severity_score"], 2),
            "actionability": round(cs.get("avg_actionability", 0), 2),
            "confidence":    round(cs.get("avg_confidence", 0), 2),
            "neg":           cs["negative_review_count"],
            "pos":           cs["positive_review_count"],
            "hsev":          cs["high_severity_count"],
            "signals":       cs.get("signals", []),
            "phrases":       cs.get("phrases", {}),
            "l1":            taxonomy.get("l1", ""),
            "l2":            taxonomy.get("l2", ""),
            "l3":            taxonomy.get("l3", ""),
            "theme":         taxonomy.get("theme", ""),
            "subtheme":      taxonomy.get("subtheme", ""),
        })

    reviews = []
    for r in pipeline_reviews:
        sent = r.get("sentiment", {})
        sev = r.get("severity", {})
        act = r.get("actionability", {})
        reviews.append({
            "id":   r["id"],
            "name": r["name"],
            "date": r["date"],
            "cid":  r.get("cluster_id", -1),
            "rep":  r.get("is_representative", False),
            "conf": round(r.get("cluster_confidence", 0), 2),
            "sent": {"c": round(sent.get("compound", 0), 2), "l": sent.get("label", "Neutral")},
            "sev":  {"s": round(sev.get("score", 0), 1), "l": sev.get("label", "Low")},
            "act":  {"s": round(act.get("score", 0), 2), "l": act.get("label", "Not actionable")},
            "text": r["text"],
        })

    return clusters, reviews
