# Emergent Review Processor

A two-in-one pipeline for processing and analysing app store reviews:

1. **Export** — Parse a `.docx` review file and export a clean, formatted `.xlsx`
2. **Cluster** — Run a full ML pipeline (embedding → UMAP → HDBSCAN) to auto-cluster,
   score, and prioritise reviews for product teams

---

## Project Structure

```
review-processor/
├── main.py                   ← Unified CLI entry point
├── requirements.txt
│
├── src/
│   ├── parsers/
│   │   └── docx_parser.py    ← Parses .docx → list of (name, date, text)
│   ├── exporters/
│   │   └── excel_exporter.py ← Writes styled .xlsx from review tuples
│   └── analysis/
│       ├── scoring.py        ← Sentiment (VADER) + severity + actionability
│       ├── embedding.py      ← sentence-transformers + UMAP reduction
│       ├── clustering.py     ← HDBSCAN + confidence scoring
│       ├── labelling.py      ← TF-IDF cluster auto-labelling
│       └── pipeline.py       ← End-to-end pipeline orchestration
│
├── data/
│   └── raw_reviews.py        ← Built-in static review dataset (fallback)
│
├── output/                   ← Generated files land here (git-ignored)
│
└── scripts/                  ← Reserved for batch / scheduled runs
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Windows note:** `hdbscan` (PyPI) requires a C++ compiler.
> This project uses `scikit-learn`'s built-in HDBSCAN (v1.3+) instead — no compiler needed.

---

## Usage

### 1. Export .docx → Excel

```bash
python main.py export --input "C:\Users\yashm\Downloads\Reviews.docx"
```

Optional custom output path:
```bash
python main.py export --input "Reviews.docx" --output "output\reviews.xlsx"
```

### 2. ML Clustering Pipeline → JSON

Using the built-in static dataset:
```bash
python -X utf8 main.py cluster
```

From a `.docx` file:
```bash
python -X utf8 main.py cluster --input "C:\Users\yashm\Downloads\Reviews.docx"
```

Custom output:
```bash
python -X utf8 main.py cluster --output "output\analysis.json"
```

---

## Outputs

| Command   | Output file                         | Contents                                      |
|-----------|--------------------------------------|-----------------------------------------------|
| `export`  | `Processed_Reviews.xlsx`            | Styled table: Name, Date, Review              |
| `cluster` | `review_analysis.json`              | Per-review scores + cluster metadata + UMAP coords |

---

## Pipeline Overview (cluster)

| Step | What happens |
|------|-------------|
| 1 | Sentiment scored via VADER (compound -1 → +1) |
| 2 | Severity scored via weighted keyword matching |
| 3 | Actionability scored by feature specificity + quantities |
| 4 | Reviews embedded: `all-MiniLM-L6-v2` → 384-dim vectors |
| 5 | UMAP: 384D → 5D (clustering) + 2D (visualisation coords) |
| 6 | HDBSCAN auto-detects clusters; labels noise as -1 |
| 7 | Blended confidence score per review |
| 8 | TF-IDF cluster auto-labelling |
| 9 | Priority score = 0.5×severity + 0.3×neg_ratio + 0.2×size |
