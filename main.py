"""
Emergent Review Processor
=========================
Single entry point for the full pipeline:

  Reviews.docx  -->  Parse  -->  ML Score + Cluster  -->  Excel Report

Usage
-----
  # Full pipeline (recommended)
  python -X utf8 main.py run

  # Specify input/output explicitly
  python -X utf8 main.py run --input "C:\\path\\to\\Reviews.docx" --output "C:\\path\\to\\Report.xlsx"

  # Save JSON intermediate output too
  python -X utf8 main.py run --save-json

Pipeline steps
--------------
  1. Parse     .docx → structured review list
  2. Score     VADER sentiment + keyword severity + actionability
  3. Embed     sentence-transformers all-MiniLM-L6-v2 (384-dim)
  4. Reduce    UMAP 384D → 5D (clustering) + 2D (viz coords)
  5. Cluster   HDBSCAN — auto-detects themes, handles noise
  6. Confidence Blended HDBSCAN probability + centroid proximity
  7. Label     TF-IDF cluster auto-labelling
  8. Export    4-sheet Excel: All Reviews | Cluster Summary | Deep-Dives | PM Dashboard
"""

import sys
import argparse
from pathlib import Path

DEFAULT_DOCX = r"C:\Users\yashm\Downloads\Reviews.docx"
DEFAULT_XLSX = r"C:\Users\yashm\Downloads\Review_Analysis.xlsx"
DEFAULT_JSON = r"C:\Users\yashm\Downloads\review_analysis.json"


# ── Full pipeline ─────────────────────────────────────────────────────────────

def cmd_run(args):
    from src.parsers             import parse_docx_reviews
    from src.analysis            import run_pipeline
    from src.exporters           import create_analysis_workbook

    input_path  = Path(args.input)
    output_xlsx = args.output
    output_json = args.json_output if args.save_json else None

    # ── Step 0: Validate input ─────────────────────────────────────────────
    if not input_path.exists():
        print(f"\nError: Input file not found:\n  {input_path}")
        print("Tip: pass --input with the correct path to your Reviews.docx\n")
        return 1

    print(f"\n{'='*60}")
    print("  EMERGENT REVIEW PROCESSOR")
    print(f"{'='*60}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_xlsx}")
    print(f"{'='*60}\n")

    # ── Step 1: Parse .docx ────────────────────────────────────────────────
    print("[1/3] Parsing document...")
    raw = parse_docx_reviews(str(input_path))
    if not raw:
        print("Error: No reviews found in the document. Check the file format.")
        return 1

    reviews = [
        {"id": i + 1, "name": name, "date": date, "text": text}
        for i, (name, date, text) in enumerate(raw)
    ]
    print(f"      Parsed {len(reviews)} reviews.\n")

    # ── Step 2: ML pipeline ────────────────────────────────────────────────
    print("[2/3] Running ML pipeline (scoring + embedding + clustering)...")
    json_path = output_json or str(Path(output_xlsx).with_suffix(".json"))
    result    = run_pipeline(reviews, output_path=json_path)

    if not args.save_json:
        # Remove the intermediate JSON unless user asked for it
        try:
            Path(json_path).unlink(missing_ok=True)
        except Exception:
            pass

    # ── Step 3: Export to Excel ────────────────────────────────────────────
    print("\n[3/3] Building Excel workbook...")
    create_analysis_workbook(result, output_path=output_xlsx)

    # ── Summary ────────────────────────────────────────────────────────────
    n_clusters = result["meta"]["n_clusters"]
    n_noise    = result["meta"]["n_noise_points"]
    top        = [c for c in result["cluster_summary"] if not c["is_noise"]][:3]

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")
    print(f"  Reviews processed : {len(reviews)}")
    print(f"  Clusters found    : {n_clusters}  (noise: {n_noise})")
    print(f"  Report saved to   : {output_xlsx}")
    if top:
        print(f"\n  Top priority clusters:")
        for i, c in enumerate(top, 1):
            print(f"    {i}. {c['cluster_label']} "
                  f"({c['review_count']} reviews, priority={c['priority_score']:.3f})")
    print(f"{'='*60}\n")

    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Emergent Review Processor — .docx → ML analysis → Excel report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -X utf8 main.py run\n"
            "  python -X utf8 main.py run --input Reviews.docx\n"
            "  python -X utf8 main.py run --input Reviews.docx --output Report.xlsx\n"
            "  python -X utf8 main.py run --save-json\n"
        ),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "run",
        help="Full pipeline: parse .docx → ML score/cluster → Excel report",
    )
    p.add_argument(
        "--input", "-i",
        default=DEFAULT_DOCX,
        metavar="PATH",
        help=f"Path to Reviews.docx  (default: {DEFAULT_DOCX})",
    )
    p.add_argument(
        "--output", "-o",
        default=DEFAULT_XLSX,
        metavar="PATH",
        help=f"Output .xlsx path  (default: {DEFAULT_XLSX})",
    )
    p.add_argument(
        "--save-json",
        action="store_true",
        default=False,
        help=f"Also save intermediate JSON to --json-output path",
    )
    p.add_argument(
        "--json-output",
        default=DEFAULT_JSON,
        metavar="PATH",
        help=f"JSON output path when --save-json is set  (default: {DEFAULT_JSON})",
    )

    return parser


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
