"""
Review Processor
================
Single entry point for the full pipeline:

  Reviews.docx  -->  Parse  -->  ML Score + Cluster  -->  Excel Report / Web Dashboard

Usage
-----
  # Full pipeline → Excel report only
  python -X utf8 main.py run

  # Launch web dashboard (runs pipeline → generates Excel + serves dashboard)
  python -X utf8 main.py web --input "C:\\path\\to\\Reviews.docx"

  # Web dashboard from pre-computed JSON (skip ML, instant)
  python -X utf8 main.py web --json review_analysis.json

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
  8. Signals   TF-IDF discriminative vocabulary + repeated phrase detection
  9. Export    4-sheet Excel | Interactive Web Dashboard
"""

import sys
import argparse
from pathlib import Path

DEFAULT_INPUT = r"C:\Users\yashm\Downloads\Processed_Reviews.xlsx"
DEFAULT_XLSX = r"C:\Users\yashm\Downloads\Review_Analysis.xlsx"
DEFAULT_JSON = r"C:\Users\yashm\Downloads\review_analysis.json"


# ── Full pipeline ─────────────────────────────────────────────────────────────

def cmd_run(args):
    from src.parsers             import parse_docx_reviews, parse_xlsx_reviews
    from src.analysis            import run_pipeline
    from src.exporters           import create_analysis_workbook

    input_path  = Path(args.input)
    output_xlsx = args.output
    output_json = args.json_output if args.save_json else None

    # ── Step 0: Validate input ─────────────────────────────────────────────
    if not input_path.exists():
        print(f"\nError: Input file not found:\n  {input_path}")
        print("Tip: pass --input with the correct path to your Reviews.docx or .xlsx\n")
        return 1

    print(f"\n{'='*60}")
    print("  REVIEW PROCESSOR")
    print(f"{'='*60}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_xlsx}")
    print(f"{'='*60}\n")

    # ── Step 1: Parse input ───────────────────────────────────────────────
    print("[1/3] Parsing document...")
    suffix = input_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        raw = parse_xlsx_reviews(str(input_path))
    else:
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


# ── Web dashboard ─────────────────────────────────────────────────────────────

def _run_pipeline_for_web(reviews: list[dict], output_xlsx: str | None = None) -> dict:
    """Run the ML pipeline and optionally generate the Excel report too."""
    from src.analysis  import run_pipeline
    from src.exporters import create_analysis_workbook
    from tempfile import NamedTemporaryFile

    json_tmp = NamedTemporaryFile(suffix=".json", delete=False).name
    pipeline_data = run_pipeline(reviews, output_path=json_tmp)
    try:
        Path(json_tmp).unlink(missing_ok=True)
    except Exception:
        pass

    if output_xlsx:
        print(f"\n  Generating Excel report → {output_xlsx}")
        create_analysis_workbook(pipeline_data, output_path=output_xlsx)

    return pipeline_data


def cmd_web(args):
    import json as _json
    import webbrowser
    from src.web.app import create_app

    pipeline_data = None

    # ── Source 1: Pre-computed JSON file ──────────────────────────────────
    if getattr(args, "json", None):
        json_path = Path(args.json)
        if not json_path.exists():
            print(f"\nError: JSON file not found: {json_path}")
            return 1

        print(f"\n{'='*60}")
        print("  REVIEW PROCESSOR — WEB (from JSON)")
        print(f"{'='*60}")
        print(f"  Loading : {json_path}\n")

        with open(json_path, "r", encoding="utf-8") as f:
            pipeline_data = _json.load(f)
        n = pipeline_data.get("meta", {}).get("total_reviews", "?")
        c = pipeline_data.get("meta", {}).get("n_clusters", "?")
        print(f"  Loaded {n} reviews, {c} clusters from JSON.\n")

    # ── Source 2: .docx file → full pipeline ─────────────────────────────
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"\nError: Input file not found:\n  {input_path}")
            print("Tip: pass --input with the correct path to your .docx or .xlsx file\n")
            return 1

        from src.parsers import parse_docx_reviews, parse_xlsx_reviews

        print(f"\n{'='*60}")
        print("  REVIEW PROCESSOR — WEB")
        print(f"{'='*60}")
        print(f"  Input : {input_path}\n")

        print("[1/3] Parsing document...")
        suffix = input_path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            raw = parse_xlsx_reviews(str(input_path))
        else:
            raw = parse_docx_reviews(str(input_path))

        if not raw:
            print("Error: No reviews found in the document. Check the file format.")
            return 1

        reviews = [
            {"id": i + 1, "name": name, "date": date, "text": text}
            for i, (name, date, text) in enumerate(raw)
        ]
        print(f"      Parsed {len(reviews)} reviews.\n")

        print("[2/3] Running ML pipeline...")
        output_xlsx = args.output
        pipeline_data = _run_pipeline_for_web(reviews, output_xlsx)
        print(f"\n[3/3] Excel report saved → {output_xlsx}")

    app = create_app(pipeline_data=pipeline_data)

    port = args.port
    url = f"http://127.0.0.1:{port}"
    print(f"\n{'='*60}")
    print(f"  Dashboard ready → {url}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    webbrowser.open(url)
    app.run(host="0.0.0.0", port=port, debug=False)
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Review Processor — .docx → ML analysis → Excel report / Web dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -X utf8 main.py run\n"
            "  python -X utf8 main.py run --input Reviews.docx --output Report.xlsx\n"
            "  python -X utf8 main.py run --save-json\n"
            "  python -X utf8 main.py web --input Reviews.docx\n"
            "  python -X utf8 main.py web --input Reviews.docx --port 8080\n"
            "  python -X utf8 main.py web --json review_analysis.json\n"
        ),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "run",
        help="Full pipeline: parse .docx → ML score/cluster → Excel report",
    )
    p.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        metavar="PATH",
        help=f"Path to .docx or .xlsx  (default: {DEFAULT_INPUT})",
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

    # ── web subcommand ────────────────────────────────────────────────────
    w = sub.add_parser(
        "web",
        help="Launch interactive web dashboard (requires --input or --json)",
    )
    w.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        metavar="PATH",
        help=f"Path to .docx or .xlsx (required unless --json)  (default: {DEFAULT_INPUT})",
    )
    w.add_argument(
        "--json", "-j",
        default=None,
        metavar="PATH",
        help="Load from a pre-computed pipeline JSON instead of running the pipeline",
    )
    w.add_argument(
        "--output", "-o",
        default=DEFAULT_XLSX,
        metavar="PATH",
        help=f"Also generate Excel report at this path  (default: {DEFAULT_XLSX})",
    )
    w.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        metavar="PORT",
        help="Port to run the web server on  (default: 5000)",
    )

    return parser


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    if args.command == "web":
        return cmd_web(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
