import json
import os
import tempfile
import unittest

from eval.eval_harness import _normalize_judge_config, _normalize_models, run_eval_from_config


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
                "label_sets": {
                    "need-thesis": {
                        "path": categories_path,
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
                "label_sets": {
                    "need-thesis": {
                        "path": categories_path,
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

    def test_enabled_judge_requires_prompt_path(self):
        with self.assertRaisesRegex(
            ValueError,
            "Judge config must include a non-empty 'prompt_path' when enabled.",
        ):
            _normalize_judge_config(
                {
                    "enabled": True,
                    "model": {"model": "judge-model"},
                },
                base_dir=".",
            )

    def test_run_eval_applies_judge_correction_to_final_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "data.json")
            categories_path = os.path.join(tmpdir, "need.json")
            prompt_path = os.path.join(tmpdir, "prompt.txt")
            judge_prompt_path = os.path.join(tmpdir, "judge.txt")
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
            with open(judge_prompt_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "JUDGE_CORRECTION\nSegments:\n${segments}\nCategories:\n${categories}\n"
                    "Candidate labels:\n${predicted_pretty}\nReturn only corrected list.\n"
                )

            config = {
                "dataset_path": dataset_path,
                "output_dir": output_dir,
                "run_name": "unit_test_judge_correction",
                "match_mode": "exact",
                "label_sets": {
                    "need-thesis": {
                        "path": categories_path,
                    }
                },
                "prompt_sets": [
                    {"name": "test", "path": prompt_path}
                ],
                "models": [
                    {
                        "name": "base-model",
                        "model": "base-model",
                        "temperature": None,
                        "reasoning_effort": None,
                        "n": 1,
                    }
                ],
                "judge": {
                    "enabled": True,
                    "mode": "correct_labels",
                    "prompt_path": judge_prompt_path,
                    "fallback_to_base_on_error": True,
                    "model": {
                        "name": "judge-model",
                        "model": "judge-model",
                        "temperature": None,
                        "reasoning_effort": None,
                        "n": 1,
                    },
                },
            }
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)

            calls = []

            def stub_call_model(**kwargs):
                calls.append(kwargs)
                if "JUDGE_CORRECTION" in kwargs["prompt"]:
                    return ["[0, 1]"]
                return ["[1, 1]"]

            result = run_eval_from_config(
                config_path,
                api_key="test-key",
                call_model_fn=stub_call_model,
                show_progress=False,
            )

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1]["model"], "judge-model")

            results_path = result["runs"][0]["results"]
            with open(results_path, "r", encoding="utf-8") as handle:
                record = json.loads(handle.readline())
            self.assertEqual(record["predicted"], [1, 1])
            self.assertEqual(record["judge_corrected"], [0, 1])
            self.assertEqual(record["final_predicted"], [0, 1])
            self.assertIsNone(record["judge_error"])
            self.assertTrue(record["exact_match"])
            self.assertEqual(record["segment_accuracy"], 1.0)

            summary_path = result["runs"][0]["summary"]
            with open(summary_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertTrue(summary["uses_final_predicted"])
            self.assertTrue(summary["judge"]["enabled"])
            self.assertEqual(summary["judge"]["examples_judge_corrected"], 1)

    def test_run_eval_judge_fallback_uses_base_prediction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "data.json")
            categories_path = os.path.join(tmpdir, "need.json")
            prompt_path = os.path.join(tmpdir, "prompt.txt")
            judge_prompt_path = os.path.join(tmpdir, "judge.txt")
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
            with open(judge_prompt_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "JUDGE_CORRECTION\nSegments:\n${segments}\nCategories:\n${categories}\n"
                    "Candidate labels:\n${predicted_pretty}\nReturn only corrected list.\n"
                )

            config = {
                "dataset_path": dataset_path,
                "output_dir": output_dir,
                "run_name": "unit_test_judge_fallback",
                "match_mode": "exact",
                "label_sets": {
                    "need-thesis": {
                        "path": categories_path,
                    }
                },
                "prompt_sets": [
                    {"name": "test", "path": prompt_path}
                ],
                "models": [
                    {
                        "name": "base-model",
                        "model": "base-model",
                        "temperature": None,
                        "reasoning_effort": None,
                        "n": 1,
                    }
                ],
                "judge": {
                    "enabled": True,
                    "mode": "correct_labels",
                    "prompt_path": judge_prompt_path,
                    "fallback_to_base_on_error": True,
                    "model": {
                        "name": "judge-model",
                        "model": "judge-model",
                        "temperature": None,
                        "reasoning_effort": None,
                        "n": 1,
                    },
                },
            }
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)

            def stub_call_model(**kwargs):
                if "JUDGE_CORRECTION" in kwargs["prompt"]:
                    return ["not a label list"]
                return ["[0, 1]"]

            result = run_eval_from_config(
                config_path,
                api_key="test-key",
                call_model_fn=stub_call_model,
                show_progress=False,
            )

            results_path = result["runs"][0]["results"]
            with open(results_path, "r", encoding="utf-8") as handle:
                record = json.loads(handle.readline())
            self.assertEqual(record["predicted"], [0, 1])
            self.assertEqual(record["judge_corrected"], [])
            self.assertEqual(record["final_predicted"], [0, 1])
            self.assertIsNotNone(record["judge_error"])
            self.assertIsNone(record["error"])
            self.assertTrue(record["exact_match"])

    def test_run_eval_can_disable_judge_via_function_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "data.json")
            categories_path = os.path.join(tmpdir, "need.json")
            prompt_path = os.path.join(tmpdir, "prompt.txt")
            judge_prompt_path = os.path.join(tmpdir, "judge.txt")
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
            with open(judge_prompt_path, "w", encoding="utf-8") as handle:
                handle.write("JUDGE_CORRECTION\n${segments}\n${categories}\n${predicted_pretty}\n")

            config = {
                "dataset_path": dataset_path,
                "output_dir": output_dir,
                "run_name": "unit_test_judge_override",
                "match_mode": "exact",
                "label_sets": {
                    "need-thesis": {
                        "path": categories_path,
                    }
                },
                "prompt_sets": [
                    {"name": "test", "path": prompt_path}
                ],
                "models": [
                    {
                        "name": "base-model",
                        "model": "base-model",
                        "temperature": None,
                        "reasoning_effort": None,
                        "n": 1,
                    }
                ],
                "judge": {
                    "enabled": True,
                    "mode": "correct_labels",
                    "prompt_path": judge_prompt_path,
                    "fallback_to_base_on_error": True,
                    "model": {
                        "name": "judge-model",
                        "model": "judge-model",
                        "temperature": None,
                        "reasoning_effort": None,
                        "n": 1,
                    },
                },
            }
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)

            calls = []

            def stub_call_model(**kwargs):
                calls.append(kwargs)
                return ["[0, 1]"]

            result = run_eval_from_config(
                config_path,
                api_key="test-key",
                call_model_fn=stub_call_model,
                show_progress=False,
                judge_enabled=False,
            )

            self.assertEqual(len(calls), 1)
            results_path = result["runs"][0]["results"]
            with open(results_path, "r", encoding="utf-8") as handle:
                record = json.loads(handle.readline())
            self.assertFalse(record["judge_enabled"])
            self.assertEqual(record["final_predicted"], [0, 1])


if __name__ == "__main__":
    unittest.main()
