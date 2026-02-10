
import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .categories import load_categories
from .prompting import DEFAULT_CATEGORIES

DEFAULT_OUTPUT_FILENAME = "eval_analysis_report.html"
DEFAULT_METRICS_FILENAME = "eval_analysis_metrics.json"
DEFAULT_MIN_WRONG_RUNS = 1
DEFAULT_MIN_WRONG_RATE = 25.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Analyze eval JSONL files and build an HTML report.")
    p.add_argument("inputs", nargs="+", help="JSONL file(s), directory path(s), or glob(s).")
    p.add_argument("--output", default=None,
                   help=f"Output HTML path (default: <input-common-dir>/{DEFAULT_OUTPUT_FILENAME}).")
    p.add_argument("--metrics-output", default=None,
                   help=f"Metrics JSON path (default: <input-common-dir>/{DEFAULT_METRICS_FILENAME}).")
    p.add_argument("--min-wrong-runs", type=int, default=DEFAULT_MIN_WRONG_RUNS,
                   help="Default report filter: include items wrong in >N runs.")
    p.add_argument("--min-wrong-rate", type=float, default=DEFAULT_MIN_WRONG_RATE,
                   help="Default report filter: include items wrong in >X percent of valid runs.")
    p.add_argument("--max-display-rows", type=int, default=250,
                   help="Max rows shown per frequent-error table.")
    return p.parse_args(argv)


def default_output_dir(files):
    if not files:
        return Path.cwd().resolve()
    return Path(os.path.commonpath([str(f.parent) for f in files])).resolve()


def resolve_output_paths(files, output_arg=None, metrics_output_arg=None):
    base_dir = default_output_dir(files)
    output = Path(output_arg).resolve() if output_arg else (base_dir / DEFAULT_OUTPUT_FILENAME)
    metrics_output = Path(metrics_output_arg).resolve() if metrics_output_arg else (base_dir / DEFAULT_METRICS_FILENAME)
    return output, metrics_output


def collect_jsonl(inputs):
    files = []
    for raw in inputs:
        p = Path(raw)
        if p.exists():
            if p.is_dir():
                files.extend(x.resolve() for x in p.rglob("*.jsonl") if x.is_file())
            elif p.is_file() and p.suffix.lower() == ".jsonl":
                files.append(p.resolve())
            continue
        for m in Path().glob(raw):
            if m.is_dir():
                files.extend(x.resolve() for x in m.rglob("*.jsonl") if x.is_file())
            elif m.is_file() and m.suffix.lower() == ".jsonl":
                files.append(m.resolve())
    return sorted(set(files))


def run_ids_for(files):
    if not files:
        return {}
    root = Path(os.path.commonpath([str(f.parent) for f in files]))
    out = {}
    for f in files:
        try:
            out[f] = str(f.relative_to(root).with_suffix("")).replace("\\", "/")
        except ValueError:
            out[f] = f.stem
    return out


def load_labels_for_jsonl(jsonl_path):
    cfg = jsonl_path.with_name(f"{jsonl_path.stem}_config.json")
    labels = {}
    warnings = []
    if not cfg.exists():
        warnings.append(f"Missing config file for label names: {cfg}")
        return labels, warnings
    try:
        payload = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"Failed to read {cfg}: {exc}")
        return labels, warnings

    label_sets = payload.get("label_sets")
    defaults = payload.get("defaults", {}) if isinstance(payload.get("defaults"), dict) else {}
    if not isinstance(label_sets, dict):
        warnings.append(f"Config missing label_sets: {cfg}")
        return labels, warnings

    for type_id, entry in label_sets.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            warnings.append(f"Invalid label set entry for {type_id} in {cfg}")
            continue
        use_defaults = bool(entry.get("use_defaults", defaults.get("use_defaults", True)))
        override_defaults = bool(entry.get("override_defaults", defaults.get("override_defaults", False)))
        label_path = Path(entry["path"])
        if not label_path.is_absolute():
            label_path = (cfg.parent / label_path).resolve()
        try:
            labels[type_id] = load_categories(
                str(label_path),
                use_defaults=use_defaults,
                override=override_defaults,
                defaults=DEFAULT_CATEGORIES,
            )
        except Exception as exc:
            warnings.append(f"Failed loading labels for {type_id} ({label_path}): {exc}")
    return labels, warnings


def name_for_label(type_id, idx, canonical):
    label = canonical.get(type_id, {}).get(idx)
    return f"{idx}: {label}" if label else str(idx)


def analyze(files, run_ids):
    canonical = defaultdict(dict)
    label_warnings = []

    for f in files:
        run = run_ids[f]
        by_type, warns = load_labels_for_jsonl(f)
        label_warnings.extend(warns)
        for type_id, labels in by_type.items():
            for i, text in enumerate(labels):
                existing = canonical[type_id].get(i)
                if existing is None:
                    canonical[type_id][i] = text
                elif existing != text:
                    label_warnings.append(
                        f"Label mismatch {type_id}[{i}] across runs: '{existing}' vs '{text}' (run {run})"
                    )

    run_stats = defaultdict(lambda: {
        "records_total": 0, "records_valid": 0, "technical_failures": 0,
        "segments_total": 0, "segments_correct": 0, "exact_matches": 0,
    })
    run_judge_stats = defaultdict(lambda: {
        "judge_enabled_records": 0,
        "judge_corrected_records": 0,
        "judge_fallback_records": 0,
        "judge_compared_records": 0,
        "judge_improved_records": 0,
        "judge_worsened_records": 0,
        "judge_unchanged_records": 0,
        "judge_segment_delta": 0,
        "judge_exact_improved_records": 0,
        "judge_exact_worsened_records": 0,
    })
    has_judge_metrics = False
    type_stats = defaultdict(lambda: {"segments_total": 0, "segments_correct": 0})
    run_type_stats = defaultdict(lambda: {"segments_total": 0, "segments_correct": 0})

    confusion = defaultdict(lambda: defaultdict(Counter))
    mislabels = defaultdict(Counter)
    expected_totals = defaultdict(Counter)

    segment_issues = {}
    example_issues = {}
    failures = []
    failure_counts = Counter()

    for f in files:
        run = run_ids[f]
        with f.open("r", encoding="utf-8") as h:
            for line_no, raw in enumerate(h, start=1):
                line = raw.strip()
                if not line:
                    continue
                run_stats[run]["records_total"] += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    reason = f"Invalid JSONL line: {exc.msg}"
                    run_stats[run]["technical_failures"] += 1
                    failure_counts[reason] += 1
                    failures.append({"run_id": run, "line_number": line_no, "example_id": None,
                                     "type_id": None, "reason": reason, "source_file": str(f)})
                    continue

                ex_id = str(rec.get("example_id", "unknown-example"))
                type_id = str(rec.get("type_id", "unknown-type"))
                expected = rec.get("expected")
                base_predicted = rec.get("predicted")
                final_predicted = rec.get("final_predicted")
                predicted = final_predicted if final_predicted is not None else base_predicted
                segments = rec.get("segments")
                error = rec.get("error")
                text = rec.get("text")
                judge_enabled = bool(rec.get("judge_enabled", False))
                judge_corrected = rec.get("judge_corrected")
                judge_error = rec.get("judge_error")

                if judge_enabled:
                    has_judge_metrics = True
                    run_judge_stats[run]["judge_enabled_records"] += 1
                    if isinstance(judge_corrected, list) and len(judge_corrected) > 0:
                        run_judge_stats[run]["judge_corrected_records"] += 1
                    elif judge_error not in (None, ""):
                        run_judge_stats[run]["judge_fallback_records"] += 1

                reason = None
                if error not in (None, ""):
                    reason = str(error)
                elif not isinstance(expected, list) or not all(isinstance(x, int) for x in expected):
                    reason = "Invalid expected label list"
                elif not isinstance(predicted, list) or not all(isinstance(x, int) for x in predicted):
                    reason = "Invalid final predicted label list"
                elif len(predicted) != len(expected):
                    reason = f"Final prediction length mismatch (predicted={len(predicted)}, expected={len(expected)})"
                elif not isinstance(segments, list):
                    reason = "Invalid segments list"
                elif len(segments) != len(expected):
                    reason = f"Segment length mismatch (segments={len(segments)}, expected={len(expected)})"

                if reason:
                    run_stats[run]["technical_failures"] += 1
                    failure_counts[reason] += 1
                    failures.append({"run_id": run, "line_number": line_no, "example_id": ex_id,
                                     "type_id": type_id, "reason": reason, "source_file": str(f)})
                    continue

                run_stats[run]["records_valid"] += 1
                total = len(expected)
                correct = 0
                base_correct = 0
                can_compare_judge = (
                    judge_enabled
                    and isinstance(base_predicted, list)
                    and len(base_predicted) == len(expected)
                    and all(isinstance(x, int) for x in base_predicted)
                )
                if can_compare_judge:
                    base_correct = sum(1 for p, e in zip(base_predicted, expected) if p == e)
                for idx, (seg_text, exp, pred) in enumerate(zip(segments, expected, predicted)):
                    expected_totals[type_id][exp] += 1
                    confusion[type_id][exp][pred] += 1
                    if exp == pred:
                        correct += 1
                    else:
                        mislabels[type_id][(exp, pred)] += 1

                    s_key = (type_id, ex_id, idx, str(seg_text))
                    s_item = segment_issues.setdefault(s_key, {
                        "type_id": type_id,
                        "example_id": ex_id,
                        "segment_index": idx,
                        "segment_text": str(seg_text),
                        "expected_label": exp,
                        "valid_runs": 0,
                        "wrong_runs": 0,
                        "wrong_run_ids": [],
                        "predicted_counts": Counter(),
                    })
                    s_item["valid_runs"] += 1
                    s_item["predicted_counts"][pred] += 1
                    if exp != pred:
                        s_item["wrong_runs"] += 1
                        s_item["wrong_run_ids"].append(run)

                run_stats[run]["segments_total"] += total
                run_stats[run]["segments_correct"] += correct
                if correct == total:
                    run_stats[run]["exact_matches"] += 1

                if can_compare_judge:
                    judge_stats = run_judge_stats[run]
                    judge_stats["judge_compared_records"] += 1
                    delta = correct - base_correct
                    judge_stats["judge_segment_delta"] += delta
                    if delta > 0:
                        judge_stats["judge_improved_records"] += 1
                    elif delta < 0:
                        judge_stats["judge_worsened_records"] += 1
                    else:
                        judge_stats["judge_unchanged_records"] += 1
                    base_exact = base_correct == total
                    final_exact = correct == total
                    if final_exact and not base_exact:
                        judge_stats["judge_exact_improved_records"] += 1
                    elif base_exact and not final_exact:
                        judge_stats["judge_exact_worsened_records"] += 1

                type_stats[type_id]["segments_total"] += total
                type_stats[type_id]["segments_correct"] += correct
                run_type_stats[(run, type_id)]["segments_total"] += total
                run_type_stats[(run, type_id)]["segments_correct"] += correct

                e_key = (type_id, ex_id)
                e_item = example_issues.setdefault(e_key, {
                    "type_id": type_id,
                    "example_id": ex_id,
                    "text": str(text) if text is not None else "",
                    "valid_runs": 0,
                    "wrong_runs": 0,
                    "wrong_run_ids": [],
                    "sum_segment_accuracy": 0.0,
                })
                e_item["valid_runs"] += 1
                acc = correct / total if total else 0.0
                e_item["sum_segment_accuracy"] += acc
                if correct < total:
                    e_item["wrong_runs"] += 1
                    e_item["wrong_run_ids"].append(run)

    run_summary = []
    for run in sorted(run_stats):
        s = run_stats[run]
        seg_acc = s["segments_correct"] / s["segments_total"] if s["segments_total"] else None
        exact = s["exact_matches"] / s["records_valid"] if s["records_valid"] else None
        row = {"run_id": run, **s, "segment_accuracy": seg_acc, "exact_match_rate": exact}
        if has_judge_metrics and run_judge_stats[run]["judge_enabled_records"] > 0:
            row.update(run_judge_stats[run])
        run_summary.append(row)

    type_summary = []
    for type_id in sorted(type_stats):
        s = type_stats[type_id]
        type_summary.append({
            "type_id": type_id,
            "segments_total": s["segments_total"],
            "segments_correct": s["segments_correct"],
            "segment_accuracy": s["segments_correct"] / s["segments_total"] if s["segments_total"] else None,
        })

    run_type_summary = []
    for (run, type_id), s in sorted(run_type_stats.items()):
        run_type_summary.append({
            "run_id": run,
            "type_id": type_id,
            "segments_total": s["segments_total"],
            "segments_correct": s["segments_correct"],
            "segment_accuracy": s["segments_correct"] / s["segments_total"] if s["segments_total"] else None,
        })

    confusion_by_type = {}
    mislabel_breakdown = {}
    for type_id in sorted(confusion):
        idxs = set(expected_totals[type_id].keys())
        for exp, preds in confusion[type_id].items():
            idxs.add(exp)
            idxs.update(preds.keys())
        idxs.update(canonical.get(type_id, {}).keys())
        idxs = sorted(idxs)

        matrix = [[confusion[type_id][e][p] for p in idxs] for e in idxs]
        confusion_by_type[type_id] = {
            "indices": idxs,
            "labels": [name_for_label(type_id, i, canonical) for i in idxs],
            "matrix": matrix,
        }

        by_expected = []
        for exp in sorted(expected_totals[type_id].keys()):
            total = expected_totals[type_id][exp]
            wrong = [(p, c) for p, c in confusion[type_id][exp].items() if p != exp and c > 0]
            wrong.sort(key=lambda x: x[1], reverse=True)
            wrong_total = sum(c for _, c in wrong)
            by_expected.append({
                "expected_label": exp,
                "expected_name": name_for_label(type_id, exp, canonical),
                "total_segments": total,
                "wrong_total": wrong_total,
                "wrong_rate": wrong_total / total if total else 0.0,
                "top_wrong_predictions": [
                    {"predicted_label": p, "predicted_name": name_for_label(type_id, p, canonical), "count": c}
                    for p, c in wrong
                ],
            })

        pairs = []
        for (exp, pred), count in mislabels[type_id].most_common():
            pairs.append({
                "expected_label": exp,
                "predicted_label": pred,
                "expected_name": name_for_label(type_id, exp, canonical),
                "predicted_name": name_for_label(type_id, pred, canonical),
                "count": count,
            })
        mislabel_breakdown[type_id] = {"by_expected": by_expected, "pairs": pairs}

    frequent_segments = []
    for s in segment_issues.values():
        valid = s["valid_runs"]
        wrong = s["wrong_runs"]
        frequent_segments.append({
            "type_id": s["type_id"],
            "example_id": s["example_id"],
            "segment_index": s["segment_index"],
            "segment_text": s["segment_text"],
            "expected_label": s["expected_label"],
            "expected_name": name_for_label(s["type_id"], s["expected_label"], canonical),
            "valid_runs": valid,
            "wrong_runs": wrong,
            "wrong_rate": wrong / valid if valid else 0.0,
            "wrong_run_ids": s["wrong_run_ids"],
            "predicted_counts": [
                {"label": i, "label_name": name_for_label(s["type_id"], i, canonical), "count": c}
                for i, c in s["predicted_counts"].most_common()
            ],
        })
    frequent_segments.sort(key=lambda x: (x["wrong_runs"], x["wrong_rate"]), reverse=True)

    frequent_examples = []
    for e in example_issues.values():
        valid = e["valid_runs"]
        wrong = e["wrong_runs"]
        frequent_examples.append({
            "type_id": e["type_id"],
            "example_id": e["example_id"],
            "text": e["text"],
            "valid_runs": valid,
            "wrong_runs": wrong,
            "wrong_rate": wrong / valid if valid else 0.0,
            "wrong_run_ids": e["wrong_run_ids"],
            "avg_segment_accuracy": e["sum_segment_accuracy"] / valid if valid else 0.0,
        })
    frequent_examples.sort(key=lambda x: (x["wrong_runs"], x["wrong_rate"]), reverse=True)

    analysis = {
        "analyzed_files": [str(f) for f in files],
        "run_summary": run_summary,
        "type_summary": type_summary,
        "run_type_summary": run_type_summary,
        "confusion_by_type": confusion_by_type,
        "mislabel_breakdown": mislabel_breakdown,
        "frequent_segments": frequent_segments,
        "frequent_examples": frequent_examples,
        "technical_failures": failures,
        "technical_failure_counts": failure_counts.most_common(),
        "label_warnings": sorted(set(label_warnings)),
    }
    if has_judge_metrics:
        total_judge_enabled = sum(run_judge_stats[r]["judge_enabled_records"] for r in run_judge_stats)
        total_judge_compared = sum(run_judge_stats[r]["judge_compared_records"] for r in run_judge_stats)
        analysis["judge_summary"] = {
            "judge_enabled_records": total_judge_enabled,
            "judge_corrected_records": sum(run_judge_stats[r]["judge_corrected_records"] for r in run_judge_stats),
            "judge_fallback_records": sum(run_judge_stats[r]["judge_fallback_records"] for r in run_judge_stats),
            "judge_compared_records": total_judge_compared,
            "judge_improved_records": sum(run_judge_stats[r]["judge_improved_records"] for r in run_judge_stats),
            "judge_worsened_records": sum(run_judge_stats[r]["judge_worsened_records"] for r in run_judge_stats),
            "judge_unchanged_records": sum(run_judge_stats[r]["judge_unchanged_records"] for r in run_judge_stats),
            "judge_segment_delta": sum(run_judge_stats[r]["judge_segment_delta"] for r in run_judge_stats),
            "judge_exact_improved_records": sum(run_judge_stats[r]["judge_exact_improved_records"] for r in run_judge_stats),
            "judge_exact_worsened_records": sum(run_judge_stats[r]["judge_exact_worsened_records"] for r in run_judge_stats),
            "judge_improved_rate": (
                sum(run_judge_stats[r]["judge_improved_records"] for r in run_judge_stats) / total_judge_compared
                if total_judge_compared else None
            ),
            "judge_worsened_rate": (
                sum(run_judge_stats[r]["judge_worsened_records"] for r in run_judge_stats) / total_judge_compared
                if total_judge_compared else None
            ),
        }
    return analysis


def h(text):
    return (
        str(text).replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")
    )


def pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"

def build_html(analysis, min_wrong_runs=None, min_wrong_rate=None, max_display_rows=250):
    total_records = sum(r["records_total"] for r in analysis["run_summary"])
    valid_records = sum(r["records_valid"] for r in analysis["run_summary"])
    total_failures = sum(r["technical_failures"] for r in analysis["run_summary"])
    total_segments = sum(r["segments_total"] for r in analysis["run_summary"])
    total_correct = sum(r["segments_correct"] for r in analysis["run_summary"])
    judge_summary = analysis.get("judge_summary")
    has_judge_metrics = judge_summary is not None
    overall_seg_acc = total_correct / total_segments if total_segments else None

    run_rows = []
    for r in analysis["run_summary"]:
        row = (
            "<tr>"
            f"<td>{h(r['run_id'])}</td>"
            f"<td>{r['records_total']}</td>"
            f"<td>{r['records_valid']}</td>"
            f"<td>{r['technical_failures']}</td>"
        )
        if has_judge_metrics:
            row += (
                f"<td>{r.get('judge_improved_records', 0)}</td>"
                f"<td>{r.get('judge_worsened_records', 0)}</td>"
                f"<td>{r.get('judge_unchanged_records', 0)}</td>"
                f"<td>{r.get('judge_segment_delta', 0)}</td>"
            )
        row += (
            f"<td>{pct(r['exact_match_rate'])}</td>"
            f"<td>{pct(r['segment_accuracy'])}</td>"
            "</tr>"
        )
        run_rows.append(row)
    run_rows = "".join(run_rows)

    type_rows = "".join(
        "<tr>"
        f"<td>{h(r['type_id'])}</td>"
        f"<td>{r['segments_total']}</td>"
        f"<td>{r['segments_correct']}</td>"
        f"<td>{pct(r['segment_accuracy'])}</td>"
        "</tr>"
        for r in analysis["type_summary"]
    )

    run_type_rows = "".join(
        "<tr>"
        f"<td>{h(r['run_id'])}</td>"
        f"<td>{h(r['type_id'])}</td>"
        f"<td>{r['segments_total']}</td>"
        f"<td>{r['segments_correct']}</td>"
        f"<td>{pct(r['segment_accuracy'])}</td>"
        "</tr>"
        for r in analysis["run_type_summary"]
    )

    confusion_sections = []
    for type_id, data in analysis["confusion_by_type"].items():
        labels = data["labels"]
        matrix = data["matrix"]
        max_cell = max((v for row in matrix for v in row), default=0)
        head = "".join(f"<th>{h(label)}</th>" for label in labels)
        body = []
        for i, row in enumerate(matrix):
            cells = [f"<th>{h(labels[i])}</th>"]
            for value in row:
                alpha = (value / max_cell) if max_cell else 0
                cells.append(f"<td style='background:rgba(37,99,235,{alpha:.3f})'>{value}</td>")
            body.append(f"<tr>{''.join(cells)}</tr>")
        confusion_sections.append(
            "<section class='panel'>"
            f"<h3>Confusion Matrix: {h(type_id)}</h3>"
            "<p class='muted'>Rows are expected labels, columns are predicted labels.</p>"
            "<div class='table-wrap'><table>"
            f"<thead><tr><th>Expected \\ Predicted</th>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div></section>"
        )

    mislabel_sections = []
    for type_id, data in analysis["mislabel_breakdown"].items():
        by_expected_rows = []
        for row in data["by_expected"]:
            top_wrong = ", ".join(
                f"{x['predicted_name']} ({x['count']})" for x in row["top_wrong_predictions"][:5]
            )
            by_expected_rows.append(
                "<tr>"
                f"<td>{h(row['expected_name'])}</td>"
                f"<td>{row['total_segments']}</td>"
                f"<td>{row['wrong_total']}</td>"
                f"<td>{row['wrong_rate'] * 100:.1f}%</td>"
                f"<td>{h(top_wrong) if top_wrong else '-'}</td>"
                "</tr>"
            )

        pair_rows = "".join(
            "<tr>"
            f"<td>{h(p['expected_name'])}</td>"
            f"<td>{h(p['predicted_name'])}</td>"
            f"<td>{p['count']}</td>"
            "</tr>"
            for p in data["pairs"][:30]
        )

        mislabel_sections.append(
            "<section class='panel'>"
            f"<h3>Mislabel Breakdown: {h(type_id)}</h3>"
            "<h4>By expected label</h4><div class='table-wrap'><table>"
            "<thead><tr><th>Expected label</th><th>Total segments</th><th>Wrong count</th><th>Wrong rate</th><th>Top wrong predictions</th></tr></thead>"
            f"<tbody>{''.join(by_expected_rows)}</tbody></table></div>"
            "<h4>Top expected -> predicted errors</h4><div class='table-wrap'><table>"
            "<thead><tr><th>Expected</th><th>Predicted</th><th>Count</th></tr></thead>"
            f"<tbody>{pair_rows}</tbody></table></div></section>"
        )

    failure_summary_rows = "".join(
        "<tr>" f"<td>{h(reason)}</td><td>{count}</td>" "</tr>"
        for reason, count in analysis["technical_failure_counts"]
    )
    failure_detail_rows = "".join(
        "<tr>"
        f"<td>{h(row['run_id'])}</td>"
        f"<td>{h(row.get('example_id') or '-')}</td>"
        f"<td>{h(row.get('type_id') or '-')}</td>"
        f"<td>{h(row['reason'])}</td>"
        f"<td>{h(row['line_number'])}</td>"
        "</tr>"
        for row in analysis["technical_failures"][:400]
    )

    warnings_html = ""
    if analysis["label_warnings"]:
        items = "".join(f"<li>{h(w)}</li>" for w in analysis["label_warnings"])
        warnings_html = f"<section class='panel'><h3>Label Resolution Warnings</h3><ul>{items}</ul></section>"

    payload = json.dumps({
        "segments": analysis["frequent_segments"],
        "examples": analysis["frequent_examples"],
        "defaults": {
            "min_wrong_runs": min_wrong_runs,
            "min_wrong_rate": min_wrong_rate,
            "max_display_rows": max_display_rows,
        },
    }, ensure_ascii=False).replace("</", "<\\/")

    judge_cards_html = ""
    judge_panel_html = ""
    run_summary_judge_headers = ""
    judge_context_note = ""
    if has_judge_metrics:
        judge_cards_html = (
            f"<div class='card'><div class='card-label'>Judge-enabled records</div><div class='card-value'>{judge_summary['judge_enabled_records']}</div></div>"
            f"<div class='card'><div class='card-label'>Judge compared records</div><div class='card-value'>{judge_summary['judge_compared_records']}</div></div>"
            f"<div class='card'><div class='card-label'>Judge improved</div><div class='card-value'>{judge_summary['judge_improved_records']}</div></div>"
            f"<div class='card'><div class='card-label'>Judge worsened</div><div class='card-value'>{judge_summary['judge_worsened_records']}</div></div>"
            f"<div class='card'><div class='card-label'>Judge segment delta</div><div class='card-value'>{judge_summary['judge_segment_delta']}</div></div>"
        )
        run_summary_judge_headers = (
            "<th>Judge improved</th><th>Judge worsened</th><th>Judge unchanged</th><th>Judge segment delta</th>"
        )
        judge_panel_html = (
            "<section class='panel'><h3>Judge Impact Summary</h3><div class='table-wrap'><table>"
            "<thead><tr><th>Compared records</th><th>Improved</th><th>Worsened</th><th>Unchanged</th><th>Improved rate</th><th>Worsened rate</th><th>Segment delta</th><th>Exact improved</th><th>Exact worsened</th></tr></thead>"
            "<tbody><tr>"
            f"<td>{judge_summary['judge_compared_records']}</td>"
            f"<td>{judge_summary['judge_improved_records']}</td>"
            f"<td>{judge_summary['judge_worsened_records']}</td>"
            f"<td>{judge_summary['judge_unchanged_records']}</td>"
            f"<td>{pct(judge_summary['judge_improved_rate'])}</td>"
            f"<td>{pct(judge_summary['judge_worsened_rate'])}</td>"
            f"<td>{judge_summary['judge_segment_delta']}</td>"
            f"<td>{judge_summary['judge_exact_improved_records']}</td>"
            f"<td>{judge_summary['judge_exact_worsened_records']}</td>"
            "</tr></tbody></table></div>"
            "<p class='muted'>Improved/worsened compares final predictions against base predictions per record.</p>"
            "</section>"
        )
        judge_context_note = (
            " Judge impact metrics compare <code>predicted</code> (base) vs <code>final_predicted</code> (after judge)."
        )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Eval Analysis Report</title>
<style>
:root{{--panel:#fff;--bg:#f8fafc;--text:#0f172a;--muted:#475569;--border:#cbd5e1}}
*{{box-sizing:border-box}} body{{margin:0;font-family:Segoe UI,Trebuchet MS,sans-serif;color:var(--text);background:radial-gradient(circle at 10% 10%,#e0f2fe,#f8fafc 35%) no-repeat}}
.container{{max-width:1300px;margin:24px auto;padding:0 16px 40px}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:16px}}
.card{{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px}} .panel{{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px;margin-bottom:14px}}
.card-label{{color:var(--muted);font-size:12px;text-transform:uppercase}} .card-value{{font-size:24px;font-weight:700;margin-top:6px}}
.table-wrap{{overflow-x:auto}} table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{border-bottom:1px solid #e2e8f0;padding:8px;text-align:left;vertical-align:top}} th{{background:#f1f5f9;white-space:nowrap}}
.muted,.small{{font-size:12px;color:var(--muted)}} .controls{{display:flex;flex-wrap:wrap;gap:10px;align-items:end;margin:10px 0 12px}}
.controls label{{display:flex;flex-direction:column;font-size:12px;color:var(--muted);gap:6px}} .controls input{{border:1px solid var(--border);border-radius:6px;padding:7px 8px;width:180px}}
.controls button{{border:1px solid #2563eb;background:#2563eb;color:#fff;border-radius:6px;padding:8px 12px;cursor:pointer}}
</style></head><body><div class='container'>
<h1>Eval Analysis Report</h1><p class='muted'>Generated {generated_at}. Mislabel metrics use <code>final_predicted</code> when present (judge-corrected), otherwise <code>predicted</code>. Technical failures are separate.{judge_context_note}</p>
<section class='cards'>
<div class='card'><div class='card-label'>JSONL files</div><div class='card-value'>{len(analysis['analyzed_files'])}</div></div>
<div class='card'><div class='card-label'>Runs</div><div class='card-value'>{len(analysis['run_summary'])}</div></div>
<div class='card'><div class='card-label'>Total records</div><div class='card-value'>{total_records}</div></div>
<div class='card'><div class='card-label'>Valid records</div><div class='card-value'>{valid_records}</div></div>
<div class='card'><div class='card-label'>Technical failures</div><div class='card-value'>{total_failures}</div></div>
{judge_cards_html}
<div class='card'><div class='card-label'>Segment accuracy</div><div class='card-value'>{pct(overall_seg_acc)}</div></div>
</section>
{judge_panel_html}
<section class='panel'><h3>Run Summary</h3><div class='table-wrap'><table><thead><tr><th>Run</th><th>Total records</th><th>Valid records</th><th>Technical failures</th>{run_summary_judge_headers}<th>Exact match rate</th><th>Segment accuracy</th></tr></thead><tbody>{run_rows}</tbody></table></div></section>
<section class='panel'><h3>Segment Match Rate by Category</h3><div class='table-wrap'><table><thead><tr><th>Category</th><th>Segments</th><th>Correct</th><th>Segment accuracy</th></tr></thead><tbody>{type_rows}</tbody></table></div></section>
<section class='panel'><h3>Segment Match Rate by Run and Category</h3><div class='table-wrap'><table><thead><tr><th>Run</th><th>Category</th><th>Segments</th><th>Correct</th><th>Segment accuracy</th></tr></thead><tbody>{run_type_rows}</tbody></table></div></section>
{''.join(confusion_sections)}
{''.join(mislabel_sections)}
<section class='panel'><h3>Often-Wrong Segments Across Runs</h3><p class='muted'>Filter with: wrong in &gt;N runs OR wrong in &gt;X% of valid runs.</p>
<div class='controls'><label>Wrong in &gt;N runs<input id='minWrongRuns' type='number' min='0' step='1' placeholder='e.g., 2'></label>
<label>Wrong in &gt;X% runs<input id='minWrongRate' type='number' min='0' max='100' step='0.1' placeholder='e.g., 50'></label><button id='applyFilter'>Apply</button><span class='small' id='segmentCount'></span></div>
<div class='table-wrap'><table><thead><tr><th>Category</th><th>Example</th><th>Segment idx</th><th>Segment text</th><th>Expected label</th><th>Wrong runs</th><th>Valid runs</th><th>Wrong rate</th><th>Predicted label counts</th></tr></thead><tbody id='segmentsBody'></tbody></table></div></section>
<section class='panel'><h3>Often-Wrong Examples Across Runs</h3><span class='small' id='exampleCount'></span><div class='table-wrap'><table><thead><tr><th>Category</th><th>Example</th><th>Wrong runs</th><th>Valid runs</th><th>Wrong rate</th><th>Avg segment accuracy</th><th>Example text</th></tr></thead><tbody id='examplesBody'></tbody></table></div></section>
<section class='panel'><h3>Technical Failures (separate from mislabels)</h3><div class='table-wrap'><table><thead><tr><th>Failure reason</th><th>Count</th></tr></thead><tbody>{failure_summary_rows}</tbody></table></div>
<h4>Sample failure rows (up to 400)</h4><div class='table-wrap'><table><thead><tr><th>Run</th><th>Example</th><th>Type</th><th>Reason</th><th>Line</th></tr></thead><tbody>{failure_detail_rows}</tbody></table></div></section>
{warnings_html}
</div>
<script>
const data={payload};
const maxRows=data.defaults.max_display_rows||250;
const minRunsInput=document.getElementById('minWrongRuns');
const minRateInput=document.getElementById('minWrongRate');
if(data.defaults.min_wrong_runs!==null&&data.defaults.min_wrong_runs!==undefined) minRunsInput.value=data.defaults.min_wrong_runs;
if(data.defaults.min_wrong_rate!==null&&data.defaults.min_wrong_rate!==undefined) minRateInput.value=data.defaults.min_wrong_rate;
function esc(t){{return String(t).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');}}
function parseNum(v){{if(v===''||v===null||v===undefined) return null; const n=Number(v); return Number.isFinite(n)?n:null;}}
function pass(item,mr,mp){{if(item.wrong_runs<=0) return false; if(mr===null&&mp===null) return true; const a=mr!==null?item.wrong_runs>mr:false; const b=mp!==null?(item.wrong_rate*100)>mp:false; return (mr!==null&&mp!==null)?(a||b):(mr!==null?a:b);}}
function render(){{
 const mr=parseNum(minRunsInput.value); const mp=parseNum(minRateInput.value);
 const seg=data.segments.filter(x=>pass(x,mr,mp)).slice(0,maxRows);
 const ex=data.examples.filter(x=>pass(x,mr,mp)).slice(0,maxRows);
 document.getElementById('segmentsBody').innerHTML=seg.map(s=>{{const preds=s.predicted_counts.map(p=>`${{esc(p.label_name)}} (${{p.count}})`).join(', '); return `<tr><td>${{esc(s.type_id)}}</td><td>${{esc(s.example_id)}}</td><td>${{s.segment_index}}</td><td>${{esc(s.segment_text)}}</td><td>${{esc(s.expected_name)}}</td><td>${{s.wrong_runs}}</td><td>${{s.valid_runs}}</td><td>${{(s.wrong_rate*100).toFixed(1)}}%</td><td>${{preds||'-'}}</td></tr>`;}}).join('');
 document.getElementById('examplesBody').innerHTML=ex.map(e=>`<tr><td>${{esc(e.type_id)}}</td><td>${{esc(e.example_id)}}</td><td>${{e.wrong_runs}}</td><td>${{e.valid_runs}}</td><td>${{(e.wrong_rate*100).toFixed(1)}}%</td><td>${{(e.avg_segment_accuracy*100).toFixed(1)}}%</td><td>${{esc(e.text)}}</td></tr>`).join('');
 document.getElementById('segmentCount').textContent=`Showing ${{seg.length}} segment rows (max ${{maxRows}} shown).`;
 document.getElementById('exampleCount').textContent=`Showing ${{ex.length}} example rows (max ${{maxRows}} shown).`;
}}
document.getElementById('applyFilter').addEventListener('click',render); render();
</script></body></html>"""

def main():
    args = parse_args()

    if args.min_wrong_runs is not None and args.min_wrong_runs < 0:
        raise ValueError("--min-wrong-runs must be non-negative")
    if args.min_wrong_rate is not None and (args.min_wrong_rate < 0 or args.min_wrong_rate > 100):
        raise ValueError("--min-wrong-rate must be between 0 and 100")
    if args.max_display_rows <= 0:
        raise ValueError("--max-display-rows must be positive")

    files = collect_jsonl(args.inputs)
    if not files:
        raise FileNotFoundError("No JSONL files found for the given inputs")

    run_ids = run_ids_for(files)
    analysis = analyze(files, run_ids)

    output, metrics_path = resolve_output_paths(
        files,
        output_arg=args.output,
        metrics_output_arg=args.metrics_output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_html(
            analysis,
            min_wrong_runs=args.min_wrong_runs,
            min_wrong_rate=args.min_wrong_rate,
            max_display_rows=args.max_display_rows,
        ),
        encoding="utf-8",
    )

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": args.inputs,
        "analyzed_files": analysis["analyzed_files"],
        "run_summary": analysis["run_summary"],
        "type_summary": analysis["type_summary"],
        "run_type_summary": analysis["run_type_summary"],
        "mislabel_breakdown": analysis["mislabel_breakdown"],
        "technical_failure_counts": analysis["technical_failure_counts"],
        "label_warnings": analysis["label_warnings"],
    }
    if "judge_summary" in analysis:
        metrics["judge_summary"] = analysis["judge_summary"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_html": str(output),
        "metrics_output": str(metrics_path),
        "jsonl_files": len(files),
        "min_wrong_runs": args.min_wrong_runs,
        "min_wrong_rate": args.min_wrong_rate,
    }, indent=2))


if __name__ == "__main__":
    main()
