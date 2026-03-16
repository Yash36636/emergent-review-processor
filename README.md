# Emergent Review Processor

A two-in-one pipeline for processing and analysing app store reviews:

1. **Excel Report** — Parse a `.docx` review file → ML pipeline → styled `.xlsx` report
2. **Web Dashboard** — Same pipeline output → interactive web dashboard for exploration

---

## Project Structure

```
review-processor/
├── main.py                   ← Unified CLI entry point (run / web)
├── app.py                    ← Vercel entry (exports Flask app)
├── vercel.json               ← Vercel config (slim requirements)
├── requirements.txt          ← Full deps (local pipeline)
├── requirements-vercel.txt   ← Slim deps (Vercel deploy)
│
├── src/
│   ├── parsers/
│   │   └── docx_parser.py    ← Parses .docx → list of (name, date, text)
│   ├── exporters/
│   │   └── excel_exporter.py ← Writes styled .xlsx from pipeline output
│   ├── analysis/
│   │   ├── scoring.py        ← Sentiment (VADER) + severity + actionability
│   │   ├── embedding.py      ← sentence-transformers + UMAP reduction
│   │   ├── clustering.py     ← HDBSCAN + confidence scoring
│   │   ├── labelling.py      ← TF-IDF cluster auto-labelling
│   │   ├── signals.py        ← TF-IDF discriminative term extraction
│   │   ├── phrases.py        ← N-gram phrase detection per cluster
│   │   ├── taxonomy.py       ← 5-level hierarchical theme mapping
│   │   └── pipeline.py       ← End-to-end pipeline orchestration
│   └── web/
│       ├── app.py            ← Flask server + API endpoints
│       └── templates/
│           └── dashboard.html← Interactive Review Intelligence dashboard
│
├── proxy/                    ← Optional Node.js AI proxy (if direct API fails)
│   ├── server.js
│   └── package.json
├── output/                   ← Generated files land here (git-ignored)
└── scripts/                  ← Reserved for batch / scheduled runs
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Windows note:** This project uses `scikit-learn`'s built-in HDBSCAN (v1.3+) — no C++ compiler needed.

---

## Usage

### 1. Excel Report Only

```bash
python -X utf8 main.py run --input "Reviews.docx"
```

Custom output path:
```bash
python -X utf8 main.py run --input "Reviews.docx" --output "output\Report.xlsx"
```

Save intermediate JSON too:
```bash
python -X utf8 main.py run --input "Reviews.docx" --save-json
```

### 2. Web Dashboard

Requires a `.docx` input (or pre-computed JSON). Runs the full pipeline, generates the Excel report, and opens the dashboard in your browser.

```bash
python -X utf8 main.py web --input "Reviews.docx"
```

From pre-computed pipeline JSON (instant, no ML):
```bash
python -X utf8 main.py web --json review_analysis.json
```

Custom port:
```bash
python -X utf8 main.py web --input "Reviews.docx" --port 8080
```

### 3. If AI API gives errors — use the Node proxy

If Inference AI chat or Generate Insight fail (e.g. 403, CORS, connection issues), run the proxy:

**Terminal 1 — start the proxy:**
```bash
cd proxy
npm install
npm start
```

**Terminal 2 — add to `.env` and run the dashboard:**
```
USE_AI_PROXY=true
AI_PROXY_URL=http://localhost:3001
```

Then run the dashboard as usual. Flask will route AI calls through the proxy instead of calling Groq directly.

---

## Vercel Deployment

Deploy the web dashboard to Vercel (serverless, no ML pipeline — dashboard + Groq API only):

1. **Connect repo** — Import the project in [Vercel](https://vercel.com/new)
2. **Environment variable** — Add `GROQ_API_KEY` in Project Settings → Environment Variables
3. **Deploy** — Vercel auto-detects Flask via `app.py` and uses `requirements-vercel.txt` (minimal deps)

After deploy, the dashboard is at `https://<project>.vercel.app`. Chat and Generate Insight use Groq directly (no Node proxy needed). The deployed app shows an empty dashboard by default; run the pipeline locally and use `main.py web --json` for data, or add a pre-computed `review_analysis.json` to the repo.

---

## Pipeline Overview

| Step | What happens |
|------|-------------|
| 1 | Parse `.docx` → structured review list |
| 2 | Sentiment scored via VADER (compound -1 → +1) |
| 3 | Severity scored via weighted keyword matching |
| 4 | Actionability scored by feature specificity + quantities |
| 5 | Reviews embedded: `all-MiniLM-L6-v2` → 384-dim vectors |
| 6 | UMAP: 384D → 5D (clustering) + 2D (visualisation coords) |
| 7 | HDBSCAN auto-detects clusters; labels noise as -1 |
| 8 | Blended confidence score per review |
| 9 | TF-IDF cluster auto-labelling |
| 10 | Discriminative signal extraction per cluster |
| 11 | N-gram phrase detection per cluster |
| 12 | 5-level taxonomy assignment |
| 13 | Priority score = 0.5×severity + 0.3×neg_ratio + 0.2×size |
