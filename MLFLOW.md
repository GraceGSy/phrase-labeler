# MLflow Logging

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
```

## What Gets Logged

- **Params:** prompt name/path, model, temperature, n, match mode, dataset path, run id
- **Metrics:** exact match rate, segment accuracy, total examples, per-type metrics
- **Artifacts:** JSONL results, summary JSON, config snapshot

## Notes

- The logger expects the eval output directory created by the harness (e.g., `eval_runs/20260127_160832_87582f45/`).
- MLflow is imported inside the logger module so the rest of the codebase does not require it.
