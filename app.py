"""
Vercel entry point — exports Flask app for serverless deployment.

Uses empty pipeline data (no ML run). Set GROQ_API_KEY in Vercel env for chat/insight.
"""
from src.web.app import create_app


def _empty_pipeline_data() -> dict:
    """Minimal pipeline structure for dashboard with no data."""
    return {
        "cluster_summary": [],
        "reviews": [],
        "meta": {
            "total_reviews": 0,
            "n_clusters": 0,
            "n_noise_points": 0,
        },
    }


app = create_app(pipeline_data=_empty_pipeline_data())
