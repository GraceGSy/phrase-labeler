import json
import os
import tempfile
import unittest

from phrase_labeler.eval_harness import _normalize_models, run_eval_from_config


class EvalHarnessTests(unittest.TestCase):
    def test_temperature_omitted_is_allowed(self):
        models = _normalize_models([
            {
                "name": "test-model",
                "model": "test-model",
                "n": 1,
            }
        ])
        self.assertIsNone(models[0]["temperature"])

    def test_reasoning_effort_null_is_allowed(self):
        models = _normalize_models([
            {
                "name": "test-model",
                "model": "test-model",
                "temperature": None,
                "reasoning_effort": None,
                "n": 1,
            }
        ])
        self.assertIsNone(models[0]["reasoning_effort"])

    def test_run_eval_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "data.json")
            categories_path = os.path.join(tmpdir, "need.json")
            prompt_path = os.path.join(tmpdir, "prompt.txt")
            output_dir = os.path.join(tmpdir, "eval_runs")
            config_path = os.path.join(tmpdir, "config.json")

            dataset = {
                "ex-1": {
                    "id": "ex-1",
                    "typeId": "need-thesis",
                    "text": "Alpha beta.",
                    "segments": {
                        "Alpha": 0,
                        "beta": 1
                    }
                }
            }
            with open(dataset_path, "w", encoding="utf-8") as handle:
                json.dump(dataset, handle)

            with open(categories_path, "w", encoding="utf-8") as handle:
                json.dump({"labels": {"0": "First", "1": "Second"}}, handle)

            with open(prompt_path, "w", encoding="utf-8") as handle:
                handle.write("Segments:\n${sentence}\nCategories:\n${categories}\n")

            config = {
                "dataset_path": dataset_path,
                "output_dir": output_dir,
                "run_name": "unit_test_run",
                "match_mode": "exact",
                "use_defaults": False,
                "override_defaults": False,
                "label_sets": {
                    "need-thesis": {
                        "path": categories_path,
                        "use_defaults": False
                    }
                },
                "prompt_sets": [
                    {"name": "test", "path": prompt_path}
                ],
                "models": [
                    {
                        "name": "test-model",
                        "model": "test-model",
                        "temperature": None,
                        "reasoning_effort": "high",
                        "n": 1,
                    }
                ]
            }
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)

            captured_kwargs = {}

            def stub_call_model(**kwargs):
                captured_kwargs.update(kwargs)
                return ["[0, 1]"]

            result = run_eval_from_config(
                config_path,
                api_key="test-key",
                call_model_fn=stub_call_model,
                show_progress=False,
            )

            self.assertEqual(result["output_dir"], output_dir)
            self.assertTrue(os.path.isdir(result["run_dir"]))
            self.assertEqual(len(result["runs"]), 1)

            summary_path = result["runs"][0]["summary"]
            self.assertTrue(os.path.exists(summary_path))

            with open(summary_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)

            self.assertEqual(captured_kwargs["reasoning_effort"], "high")
            self.assertEqual(summary["exact_match_rate"], 1.0)
            self.assertEqual(summary["segment_accuracy"], 1.0)
            self.assertEqual(summary["reasoning_effort"], "high")

    def test_invalid_reasoning_effort_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "data.json")
            categories_path = os.path.join(tmpdir, "need.json")
            prompt_path = os.path.join(tmpdir, "prompt.txt")
            config_path = os.path.join(tmpdir, "config.json")

            dataset = {
                "ex-1": {
                    "id": "ex-1",
                    "typeId": "need-thesis",
                    "text": "Alpha beta.",
                    "segments": {
                        "Alpha": 0,
                        "beta": 1
                    }
                }
            }
            with open(dataset_path, "w", encoding="utf-8") as handle:
                json.dump(dataset, handle)

            with open(categories_path, "w", encoding="utf-8") as handle:
                json.dump({"labels": {"0": "First", "1": "Second"}}, handle)

            with open(prompt_path, "w", encoding="utf-8") as handle:
                handle.write("Segments:\n${sentence}\nCategories:\n${categories}\n")

            config = {
                "dataset_path": dataset_path,
                "output_dir": os.path.join(tmpdir, "eval_runs"),
                "run_name": "unit_test_run",
                "match_mode": "exact",
                "use_defaults": False,
                "override_defaults": False,
                "label_sets": {
                    "need-thesis": {
                        "path": categories_path,
                        "use_defaults": False
                    }
                },
                "prompt_sets": [
                    {"name": "test", "path": prompt_path}
                ],
                "models": [
                    {
                        "name": "test-model",
                        "model": "test-model",
                        "temperature": None,
                        "reasoning_effort": "ultra",
                        "n": 1,
                    }
                ]
            }
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)

            with self.assertRaisesRegex(
                ValueError,
                "reasoning_effort must be null or one of: low, medium, high, xhigh",
            ):
                run_eval_from_config(
                    config_path,
                    api_key="test-key",
                    call_model_fn=lambda **_kwargs: ["[0, 1]"],
                    show_progress=False,
                )

    def test_legacy_reasoning_field_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            "Model config field 'reasoning' is not supported. Use 'reasoning_effort' instead.",
        ):
            _normalize_models([
                {
                    "name": "test-model",
                    "model": "test-model",
                    "temperature": None,
                    "reasoning": {"effort": "high"},
                    "n": 1,
                }
            ])


if __name__ == "__main__":
    unittest.main()
