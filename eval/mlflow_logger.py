import json
import os
from typing import Dict, Iterable, Optional


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _flatten_metrics(prefix: str, metrics: Dict) -> Dict[str, float]:
    flattened = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, (int, float)):
            flattened[name] = float(value)
        elif isinstance(value, dict):
            flattened.update(_flatten_metrics(f"{name}.", value))
    return flattened


def _list_artifacts(run_dir: str, base_name: str) -> list[str]:
    artifacts = []
    for suffix in (".jsonl", "_summary.json", "_config.json"):
        path = os.path.join(run_dir, f"{base_name}{suffix}")
        if os.path.exists(path):
            artifacts.append(path)
    return artifacts


def _discover_runs(run_dir: str) -> list[Dict[str, str]]:
    runs = []
    for filename in os.listdir(run_dir):
        if not filename.endswith("_summary.json"):
            continue
        base_name = filename[:-len("_summary.json")]
        summary_path = os.path.join(run_dir, filename)
        config_path = os.path.join(run_dir, f"{base_name}_config.json")
        jsonl_path = os.path.join(run_dir, f"{base_name}.jsonl")
        runs.append({
            "base_name": base_name,
            "summary": summary_path,
            "config": config_path if os.path.exists(config_path) else None,
            "jsonl": jsonl_path if os.path.exists(jsonl_path) else None,
        })
    if not runs:
        raise FileNotFoundError(f"No *_summary.json files found in {run_dir}.")
    return runs


def _iter_jsonl(path: Optional[str]) -> Iterable[dict]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def log_run_dir_to_mlflow(
    run_dir: str,
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    run_name_prefix: Optional[str] = None,
    per_example_metrics: bool = True,
) -> list[str]:
    import mlflow

    run_dir = os.path.abspath(run_dir)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    run_ids = []
    for entry in _discover_runs(run_dir):
        summary = _load_json(entry["summary"])
        config = _load_json(entry["config"]) if entry["config"] else {}

        run_name_parts = []
        if run_name_prefix:
            run_name_parts.append(run_name_prefix)
        if summary.get("run_id"):
            run_name_parts.append(summary["run_id"])
        if summary.get("prompt_name"):
            run_name_parts.append(str(summary["prompt_name"]))
        if summary.get("model"):
            run_name_parts.append(str(summary["model"]))
        run_name = " / ".join(run_name_parts) if run_name_parts else entry["base_name"]

        with mlflow.start_run(run_name=run_name) as run:
            params = {
                "prompt_name": summary.get("prompt_name"),
                "prompt_file": summary.get("prompt_file"),
                "model": summary.get("model"),
                "temperature": summary.get("temperature"),
                "reasoning_effort": summary.get("reasoning_effort"),
                "n": summary.get("n"),
                "match_mode": summary.get("match_mode"),
                "dataset_path": summary.get("dataset_path"),
                "run_id": summary.get("run_id"),
            }
            for key, value in list(params.items()):
                if value is None:
                    params.pop(key, None)
            mlflow.log_params(params)

            if config:
                mlflow.log_param("config_path", config.get("config_path"))
                mlflow.log_param("run_dir", config.get("run_dir"))

            metrics = {
                "exact_match_rate": summary.get("exact_match_rate"),
                "segment_accuracy": summary.get("segment_accuracy"),
                "total_examples": summary.get("total_examples"),
            }
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})

            by_type = summary.get("by_type", {})
            mlflow.log_metrics(_flatten_metrics("by_type.", by_type))

            if per_example_metrics:
                for record in _iter_jsonl(entry.get("jsonl")):
                    example_id = record.get("example_id")
                    if not example_id:
                        continue
                    exact_match = record.get("exact_match")
                    segment_accuracy = record.get("segment_accuracy")
                    metrics = {}
                    if exact_match is not None:
                        metrics[f"example.{example_id}.exact_match"] = 1.0 if exact_match else 0.0
                    if segment_accuracy is not None:
                        metrics[f"example.{example_id}.segment_accuracy"] = float(segment_accuracy)
                    if metrics:
                        mlflow.log_metrics(metrics)

            for artifact in _list_artifacts(run_dir, entry["base_name"]):
                mlflow.log_artifact(artifact)

            run_ids.append(run.info.run_id)
    return run_ids
