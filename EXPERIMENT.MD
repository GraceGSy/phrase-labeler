# Experiment Tracking (MLflow)

This repo includes a separate script that logs completed eval runs to MLflow.

## Install

```bash
pip install -e ".[eval,mlflow]"
```

## Start MLflow UI (local)

```bash
mlflow ui
```

Then open the UI at `http://127.0.0.1:5000`.

## Log a Run Directory

```bash
python scripts/log_to_mlflow.py --run-dir eval_runs/<timestamp_dir>
```

Optional flags:

```bash
--experiment-name "Phrase Labeler"
--tracking-uri "http://127.0.0.1:5000"
--run-name-prefix "baseline"
--no-per-example-metrics
```

## What Gets Logged

- **Params:** prompt name/path, model, temperature, n, match mode, dataset path, run id
- **Metrics:** exact match rate, segment accuracy, total examples, per-type metrics
- **Per-example metrics:** `example.<id>.exact_match` and `example.<id>.segment_accuracy` (enabled by default)
- **Artifacts:** JSONL results, summary JSON, config snapshot

## Notes

- The logger expects the eval output directory created by the harness (e.g., `eval_runs/20260127_160832_87582f45/`).
- MLflow is imported inside the logger module so the rest of the codebase does not require it.

## Comparing Multiple Prompt Sets

1) Add multiple prompt entries under `prompt_sets` in your eval config:

```json
"prompt_sets": [
  { "name": "default", "path": "prompts/default.txt" },
  { "name": "test-1", "path": "prompts/prompt-test-1.txt" }
]
```

2) Set a `run_name` in your config and run the eval once:

```bash
python scripts/run_eval.py --config eval_config.json
```

This creates a single run folder that contains one JSONL + summary per prompt/model combination, for example:

```
eval_runs/baseline_eval/
  default_gpt-41-mini_t0.2.jsonl
  default_gpt-41-mini_t0.2_summary.json
  default_gpt-41-mini_t0.2_config.json
  test-1_gpt-41-mini_t0.2.jsonl
  test-1_gpt-41-mini_t0.2_summary.json
  test-1_gpt-41-mini_t0.2_config.json
```

3) Log that run folder to MLflow:

```bash
python scripts/log_to_mlflow.py --run-dir eval_runs/20260127_160832_87582f45
```

MLflow creates a separate run for each summary file, so prompt variants are comparable in the UI.

## About Timestamps

Each eval execution produces **one** timestamped folder. Having multiple prompt sets in the same folder does **not** cause conflicts because each prompt/model combo writes its own files and MLflow creates separate runs when you log the folder.
