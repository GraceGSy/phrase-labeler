import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .analyze_eval_results import (
    DEFAULT_MIN_WRONG_RATE,
    DEFAULT_MIN_WRONG_RUNS,
    analyze,
    collect_jsonl,
    default_output_dir,
    run_ids_for,
)

DEFAULT_OUTPUT_FILENAME = "often_wrong_segments.csv"
DEFAULT_MAX_DISPLAY_ROWS = 250


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Export often-wrong segments to CSV from eval JSONL files. "
            "Filters match the HTML report behavior."
        )
    )
    p.add_argument("inputs", nargs="+", help="JSONL file(s), directory path(s), or glob(s).")
    p.add_argument(
        "--output",
        default=None,
        help=f"Output CSV path (default: <input-common-dir>/{DEFAULT_OUTPUT_FILENAME}).",
    )
    p.add_argument(
        "--min-wrong-runs",
        type=int,
        default=DEFAULT_MIN_WRONG_RUNS,
        help="Include items wrong in >N runs.",
    )
    p.add_argument(
        "--min-wrong-rate",
        type=float,
        default=DEFAULT_MIN_WRONG_RATE,
        help="Include items wrong in >X percent of valid runs.",
    )
    p.add_argument(
        "--max-display-rows",
        type=int,
        default=DEFAULT_MAX_DISPLAY_ROWS,
        help="Max rows exported (matches HTML report default behavior).",
    )
    return p.parse_args(argv)


def resolve_output_path(files, output_arg=None):
    base_dir = default_output_dir(files)
    return Path(output_arg).resolve() if output_arg else (base_dir / DEFAULT_OUTPUT_FILENAME)


def passes_filter(item, min_wrong_runs=None, min_wrong_rate=None):
    if item["wrong_runs"] <= 0:
        return False
    if min_wrong_runs is None and min_wrong_rate is None:
        return True
    runs_ok = item["wrong_runs"] > min_wrong_runs if min_wrong_runs is not None else False
    rate_ok = (item["wrong_rate"] * 100) > min_wrong_rate if min_wrong_rate is not None else False
    if min_wrong_runs is not None and min_wrong_rate is not None:
        return runs_ok or rate_ok
    return runs_ok if min_wrong_runs is not None else rate_ok


def build_rows(analysis, min_wrong_runs=None, min_wrong_rate=None, max_rows=DEFAULT_MAX_DISPLAY_ROWS):
    text_by_example = {
        (row["type_id"], row["example_id"]): row["text"]
        for row in analysis["frequent_examples"]
    }

    rows = []
    for row in analysis["frequent_segments"]:
        if not passes_filter(row, min_wrong_runs=min_wrong_runs, min_wrong_rate=min_wrong_rate):
            continue
        wrong_only = [x for x in row["predicted_counts"] if x["label"] != row["expected_label"]]
        wrong_counts = ", ".join(f"{x['label_name']} ({x['count']})" for x in wrong_only)
        rows.append({
            "data_label": row["example_id"],
            "sentence_text": text_by_example.get((row["type_id"], row["example_id"]), ""),
            "mislabeled_fragment": row["segment_text"],
            "expected_label": row["expected_name"],
            "incorrect_prediction_counts": wrong_counts if wrong_counts else "-",
            "wrong_rate_percent": round(row["wrong_rate"] * 100, 1),
        })

    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "data_label",
        "sentence_text",
        "mislabeled_fragment",
        "expected_label",
        "incorrect_prediction_counts",
        "wrong_rate_percent",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)

    if args.min_wrong_runs is not None and args.min_wrong_runs < 0:
        raise ValueError("--min-wrong-runs must be non-negative")
    if args.min_wrong_rate is not None and (args.min_wrong_rate < 0 or args.min_wrong_rate > 100):
        raise ValueError("--min-wrong-rate must be between 0 and 100")
    if args.max_display_rows <= 0:
        raise ValueError("--max-display-rows must be positive")

    files = collect_jsonl(args.inputs)
    if not files:
        raise FileNotFoundError("No JSONL files found for the given inputs")

    analysis = analyze(files, run_ids_for(files))
    rows = build_rows(
        analysis,
        min_wrong_runs=args.min_wrong_runs,
        min_wrong_rate=args.min_wrong_rate,
        max_rows=args.max_display_rows,
    )

    output = resolve_output_path(files, output_arg=args.output)
    write_csv(rows, output)

    print(json.dumps({
        "output_csv": str(output),
        "rows_written": len(rows),
        "jsonl_files": len(files),
        "min_wrong_runs": args.min_wrong_runs,
        "min_wrong_rate": args.min_wrong_rate,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


if __name__ == "__main__":
    main()
