"""
Shared LLM helper for Groq API calls.

Uses urllib.request directly with GROQ_API_KEY from .env.
"""

import json
import os
import re
import urllib.request
import urllib.error

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_MODEL = "llama-3.3-70b-versatile"


def call_groq(prompt, max_tokens=600):
    """Call Groq API and return raw text response. Strips markdown fences."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to .env.")

    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    })
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload.encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
            "User-Agent": "ReviewIntelPipeline/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read())

    raw = data["choices"][0]["message"]["content"].strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw
