import tempfile
import unittest
from pathlib import Path

from phrase_labeler.analyze_eval_results import (
    DEFAULT_METRICS_FILENAME,
    DEFAULT_MIN_WRONG_RATE,
    DEFAULT_MIN_WRONG_RUNS,
    DEFAULT_OUTPUT_FILENAME,
    analyze,
    build_html,
    collect_jsonl,
    parse_args,
    run_ids_for,
    resolve_output_paths,
)


class AnalyzeEvalCLITests(unittest.TestCase):
    def test_parse_args_defaults(self):
        args = parse_args(["eval_runs"])
        self.assertIsNone(args.output)
        self.assertIsNone(args.metrics_output)
        self.assertEqual(args.min_wrong_runs, DEFAULT_MIN_WRONG_RUNS)
        self.assertEqual(args.min_wrong_rate, DEFAULT_MIN_WRONG_RATE)

    def test_resolve_output_paths_defaults_to_common_input_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_a = root / "run_a"
            run_b = root / "run_b"
            run_a.mkdir(parents=True, exist_ok=True)
            run_b.mkdir(parents=True, exist_ok=True)
            (run_a / "a.jsonl").write_text("{}", encoding="utf-8")
            (run_b / "b.jsonl").write_text("{}", encoding="utf-8")

            files = collect_jsonl([str(root)])
            output, metrics = resolve_output_paths(files)

            self.assertEqual(output, (root / DEFAULT_OUTPUT_FILENAME).resolve())
            self.assertEqual(metrics, (root / DEFAULT_METRICS_FILENAME).resolve())

    def test_resolve_output_paths_respects_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = root / "run"
            run.mkdir(parents=True, exist_ok=True)
            (run / "a.jsonl").write_text("{}", encoding="utf-8")
            files = collect_jsonl([str(run)])

            out_path = root / "custom" / "report.html"
            metrics_path = root / "custom" / "metrics.json"
            output, metrics = resolve_output_paths(
                files,
                output_arg=str(out_path),
                metrics_output_arg=str(metrics_path),
            )

            self.assertEqual(output, out_path.resolve())
            self.assertEqual(metrics, metrics_path.resolve())

    def test_analyze_without_judge_hides_judge_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = root / "run"
            run.mkdir(parents=True, exist_ok=True)
            results = run / "sample.jsonl"
            results.write_text(
                (
                    '{"example_id":"ex-1","type_id":"need-thesis","segments":["A","B"],'
                    '"expected":[0,1],"predicted":[0,1],"final_predicted":[0,1],"error":null}\n'
                ),
                encoding="utf-8",
            )

            files = collect_jsonl([str(run)])
            analysis = analyze(files, run_ids_for(files))
            self.assertNotIn("judge_summary", analysis)
            self.assertNotIn("judge_improved_records", analysis["run_summary"][0])

            html = build_html(analysis, min_wrong_runs=1, min_wrong_rate=25.0)
            self.assertNotIn("Judge Impact Summary", html)
            self.assertNotIn("Judge-enabled records", html)

    def test_analyze_with_judge_reports_improvements_and_regressions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = root / "run"
            run.mkdir(parents=True, exist_ok=True)
            results = run / "sample.jsonl"
            results.write_text(
                "\n".join([
                    (
                        '{"example_id":"ex-1","type_id":"need-thesis","segments":["A","B"],'
                        '"expected":[0,1],"predicted":[1,1],"final_predicted":[0,1],'
                        '"judge_enabled":true,"judge_corrected":[0,1],"judge_error":null,"error":null}'
                    ),
                    (
                        '{"example_id":"ex-2","type_id":"need-thesis","segments":["A","B"],'
                        '"expected":[0,1],"predicted":[0,1],"final_predicted":[1,1],'
                        '"judge_enabled":true,"judge_corrected":[1,1],"judge_error":null,"error":null}'
                    ),
                    (
                        '{"example_id":"ex-3","type_id":"need-thesis","segments":["A","B"],'
                        '"expected":[0,1],"predicted":[0,1],"final_predicted":[0,1],'
                        '"judge_enabled":true,"judge_corrected":[0,1],"judge_error":null,"error":null}'
                    ),
                ]) + "\n",
                encoding="utf-8",
            )

            files = collect_jsonl([str(run)])
            analysis = analyze(files, run_ids_for(files))
            judge_summary = analysis.get("judge_summary")
            self.assertIsNotNone(judge_summary)
            self.assertEqual(judge_summary["judge_enabled_records"], 3)
            self.assertEqual(judge_summary["judge_compared_records"], 3)
            self.assertEqual(judge_summary["judge_improved_records"], 1)
            self.assertEqual(judge_summary["judge_worsened_records"], 1)
            self.assertEqual(judge_summary["judge_unchanged_records"], 1)
            self.assertEqual(judge_summary["judge_segment_delta"], 0)
            self.assertEqual(judge_summary["judge_exact_improved_records"], 1)
            self.assertEqual(judge_summary["judge_exact_worsened_records"], 1)
            self.assertAlmostEqual(judge_summary["judge_improved_rate"], 1 / 3)
            self.assertAlmostEqual(judge_summary["judge_worsened_rate"], 1 / 3)

            run_row = analysis["run_summary"][0]
            self.assertEqual(run_row["judge_improved_records"], 1)
            self.assertEqual(run_row["judge_worsened_records"], 1)
            self.assertEqual(run_row["judge_unchanged_records"], 1)

            html = build_html(analysis, min_wrong_runs=1, min_wrong_rate=25.0)
            self.assertIn("Judge Impact Summary", html)
            self.assertIn("Judge-enabled records", html)


if __name__ == "__main__":
    unittest.main()
