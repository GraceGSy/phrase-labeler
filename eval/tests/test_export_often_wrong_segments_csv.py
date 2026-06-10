import csv
import json
import tempfile
import unittest
from pathlib import Path

from eval.analyze_eval_results import analyze, collect_jsonl, run_ids_for
from eval.export_often_wrong_segments_csv import (
    build_rows,
    passes_filter,
    resolve_output_path,
    write_csv,
)


class ExportOftenWrongSegmentsCsvTests(unittest.TestCase):
    def test_build_rows_includes_sentence_and_wrong_only_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels_path = root / "labels.json"
            labels_path.write_text(
                json.dumps(["Stakeholders", "Setting", "Goal", "Obstacle", "Constraints"]),
                encoding="utf-8",
            )

            for run_name, pred in [("run_a", 4), ("run_b", 2)]:
                run_dir = root / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                jsonl_path = run_dir / "sample.jsonl"
                record = {
                    "example_id": "need-1",
                    "type_id": "need-thesis",
                    "text": "Users struggle with visualization editing.",
                    "segments": ["with visualization editing"],
                    "expected": [2],
                    "predicted": [pred],
                    "final_predicted": [pred],
                    "error": None,
                }
                jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

                config = {
                    "defaults": {"use_defaults": False, "override_defaults": False},
                    "label_sets": {
                        "need-thesis": {
                            "path": str(labels_path),
                            "use_defaults": False,
                            "override_defaults": False,
                        }
                    },
                }
                (run_dir / "sample_config.json").write_text(json.dumps(config), encoding="utf-8")

            files = collect_jsonl([str(root)])
            analysis = analyze(files, run_ids_for(files))
            rows = build_rows(analysis, min_wrong_runs=0, min_wrong_rate=0, max_rows=250)

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["data_label"], "need-1")
            self.assertEqual(row["sentence_text"], "Users struggle with visualization editing.")
            self.assertEqual(row["mislabeled_fragment"], "with visualization editing")
            self.assertEqual(row["expected_label"], "2: Goal")
            self.assertEqual(row["incorrect_prediction_counts"], "4: Constraints (1)")
            self.assertEqual(row["wrong_rate_percent"], 50.0)

    def test_passes_filter_matches_html_logic(self):
        row = {"wrong_runs": 1, "wrong_rate": 0.5}
        self.assertTrue(passes_filter(row, None, None))
        self.assertTrue(passes_filter(row, 0, None))
        self.assertFalse(passes_filter(row, 1, None))
        self.assertTrue(passes_filter(row, None, 25))
        self.assertFalse(passes_filter(row, None, 50))
        self.assertTrue(passes_filter(row, 1, 25))
        self.assertFalse(passes_filter(row, 1, 60))

    def test_write_csv_and_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            file_path = run_dir / "sample.jsonl"
            file_path.write_text("{}", encoding="utf-8")

            output = resolve_output_path([file_path], output_arg=None)
            rows = [{
                "data_label": "need-1",
                "sentence_text": "S",
                "mislabeled_fragment": "F",
                "expected_label": "2: Goal",
                "incorrect_prediction_counts": "4: Constraints (1)",
                "wrong_rate_percent": 50.0,
            }]
            write_csv(rows, output)

            with output.open("r", encoding="utf-8", newline="") as handle:
                parsed = list(csv.DictReader(handle))
            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0]["data_label"], "need-1")


if __name__ == "__main__":
    unittest.main()
