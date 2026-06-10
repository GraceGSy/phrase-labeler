# Eval Tools

Research and evaluation tooling for the `phrase-labeler` package. These scripts are **not part of the pip package** — they live in `eval/` and are intended for use within this repository only.

## Prerequisites

The core package must be installed (editable mode is recommended for development):

```bash
pip install -e .
pip install tqdm          # required for progress bars
pip install mlflow        # optional, only needed for MLflow logging
```

`OPENAI_API_KEY` must be set in your environment or in a `.env` file at the repo root.

## Dataset format

The dataset JSON is a dict keyed by example ID:

```json
{
  "ex-1": {
    "id": "ex-1",
    "typeId": "need-thesis",
    "text": "Full sentence text.",
    "segments": {
      "segment one": 0,
      "segment two": 2
    }
  }
}
```

- `typeId` — determines which label set is used (must match a key in `label_sets` of the eval config)
- `segments` — map of segment text → expected label index

## Eval config

Copy `eval_config.example.json` to `eval_config.json` and fill in your paths.

```json
{
  "dataset_path": "tests/data.json",
  "output_dir": "eval_runs",
  "run_name": "baseline_eval",
  "match_mode": "exact",
  "label_sets": {
    "need-thesis": { "path": "categories/need.json" },
    "approach-thesis": { "path": "categories/approach.json" }
  },
  "prompt_sets": [
    { "name": "default", "path": "prompts/default.txt" }
  ],
  "models": [
    {
      "name": "gpt-4.1-mini",
      "model": "gpt-4.1-mini",
      "temperature": 0.2,
      "reasoning_effort": null,
      "n": 1
    }
  ],
  "judge": {
    "enabled": false,
    "mode": "correct_labels",
    "prompt_path": "prompts/judge/correct-labels.txt",
    "fallback_to_base_on_error": true,
    "model": {
      "name": "gpt-4.1-mini-judge",
      "model": "gpt-4.1-mini",
      "temperature": null,
      "reasoning_effort": null,
      "n": 1
    }
  }
}
```

**Key fields:**

| Field | Description |
|---|---|
| `match_mode` | `"exact"` (full sequence match) or `"segment"` (per-segment accuracy) |
| `label_sets` | Map of `typeId` → `{ "path": "..." }`. Path is relative to the config file. |
| `prompt_sets` | List of `{ "name", "path" }`. Each prompt is crossed with each model. |
| `models[].reasoning_effort` | `null`, `"low"`, `"medium"`, `"high"`, or `"xhigh"` |
| `models[].n` | Number of completions to request per example |
| `judge.enabled` | When `true`, the judge model re-labels each prediction and the corrected label becomes `final_predicted` |
| `judge.fallback_to_base_on_error` | If the judge returns unparseable output, fall back to the base prediction |

Paths in the config are resolved relative to the config file's directory.

## Running an eval

```bash
python eval/run_eval.py --config eval_config.json --api-key YOUR_KEY
```

Arguments:

| Flag | Description |
|---|---|
| `--config` | Path to eval config JSON (required) |
| `--api-key` | OpenAI API key (overrides env var) |
| `--api-key-env` | Environment variable to read the key from (default: `OPENAI_API_KEY`) |
| `--judge-enabled` / `--no-judge-enabled` | Override `judge.enabled` from config |
| `--experiment-name` | If set, log the run to MLflow under this experiment name |
| `--tracking-uri` | MLflow tracking URI (optional) |

Output is written to `<output_dir>/<run_name>/`. Each prompt × model combination produces:
- `<name>.jsonl` — one JSON record per example with `predicted`, `final_predicted`, `exact_match`, `segment_accuracy`, etc.
- `<name>_summary.json` — aggregate metrics for the run

## Analyzing results

Generate an HTML report and metrics JSON from one or more run directories:

```bash
python -m eval.analyze_eval_results eval_runs/my-run/

# Across multiple runs:
python -m eval.analyze_eval_results eval_runs/

# Override output location and filters:
python -m eval.analyze_eval_results eval_runs/my-run/ \
  --output report.html \
  --metrics-output metrics.json \
  --min-wrong-runs 2 \
  --min-wrong-rate 50
```

**Default filter:** items wrong in `> 1` run **or** `> 25%` of valid runs are highlighted. Adjust with `--min-wrong-runs` and `--min-wrong-rate`.

Outputs:
- `eval_analysis_report.html` — interactive HTML report with per-segment error breakdown
- `eval_analysis_metrics.json` — machine-readable aggregate metrics

## Exporting often-wrong segments to CSV

Export the frequently mislabeled segments to a spreadsheet-friendly CSV:

```bash
python -m eval.export_often_wrong_segments_csv eval_runs/my-run/

# With filters:
python -m eval.export_often_wrong_segments_csv eval_runs/ \
  --output often_wrong.csv \
  --min-wrong-runs 2 \
  --min-wrong-rate 50 \
  --max-display-rows 500
```

The output CSV has columns: `data_label`, `sentence_text`, `mislabeled_fragment`, `expected_label`, `incorrect_prediction_counts`, `wrong_rate_percent`.

## Logging to MLflow

Log a completed eval run to an MLflow experiment:

```bash
python eval/log_to_mlflow.py \
  --run-dir eval_runs/my-run/ \
  --experiment-name my-experiment \
  --tracking-uri http://localhost:5000   # optional; omit to use local mlruns/
```

This can also be triggered inline during `run_eval.py` via `--experiment-name`.

Arguments:

| Flag | Description |
|---|---|
| `--run-dir` | Path to a single eval run directory (required) |
| `--experiment-name` | MLflow experiment name |
| `--tracking-uri` | MLflow tracking URI |
| `--run-name-prefix` | Prefix prepended to MLflow run names |
| `--no-per-example-metrics` | Skip logging per-example metrics (faster) |

## Output record format (JSONL)

Each line in a results `.jsonl` file is a JSON object with:

```json
{
  "example_id": "ex-1",
  "type_id": "need-thesis",
  "text": "Full sentence text.",
  "segments": ["segment one", "segment two"],
  "expected": [0, 2],
  "predicted": [0, 1],
  "final_predicted": [0, 2],
  "exact_match": true,
  "segment_accuracy": 1.0,
  "judge_enabled": true,
  "judge_corrected": [0, 2],
  "judge_error": null,
  "error": null
}
```

`final_predicted` equals `judge_corrected` when the judge is enabled and succeeds, otherwise equals `predicted`.

## Running tests

```bash
python -m unittest discover -s eval/tests -p "test_*.py" -v

# Or run the full suite (core + eval) from the repo root:
python -m unittest discover -s . -p "test_*.py" -v
```
