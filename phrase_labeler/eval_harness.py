import ast
import json
import os
import uuid
from datetime import datetime
from typing import Callable, Dict, Iterable, Optional

from .categories import load_categories
from .prompting import DEFAULT_CATEGORIES, build_prompt


def _resolve_path(base_dir: str, path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines).strip()


def _extract_bracket_list(text: str) -> Optional[str]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_labels(raw_text: str) -> list[int]:
    cleaned = _strip_code_fences(raw_text)
    candidates = [cleaned, _extract_bracket_list(cleaned)]
    for candidate in candidates:
        if not candidate:
            continue
        for parser in (json.loads, ast.literal_eval):
            try:
                data = parser(candidate)
            except (ValueError, SyntaxError):
                continue
            if isinstance(data, list) and all(isinstance(i, int) for i in data):
                return data
    raise ValueError("Could not parse label list from response.")


def call_model(
    prompt: str,
    model: str,
    temperature: float,
    n: int,
    api_key: str,
    system_prompt: Optional[str] = None,
) -> list[str]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        n=n,
        temperature=temperature,
    )
    return [choice.message.content for choice in response.choices]


def _slugify(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum() or char in ("-", "_"):
            cleaned.append(char)
        elif char.isspace():
            cleaned.append("-")
    return "".join(cleaned) or "run"


def _normalize_prompt_sets(prompt_sets: Iterable, base_dir: str) -> list[dict]:
    normalized = []
    for entry in prompt_sets:
        if isinstance(entry, str):
            path = _resolve_path(base_dir, entry)
            normalized.append({"name": os.path.basename(entry), "path": path})
            continue
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError("Each prompt set must be a string path or an object with a 'path' field.")
        name = entry.get("name") or os.path.basename(entry["path"])
        normalized.append({
            "name": name,
            "path": _resolve_path(base_dir, entry["path"]),
            "system_prompt": entry.get("system_prompt"),
        })
    return normalized


def _normalize_models(models: Iterable) -> list[dict]:
    normalized = []
    for entry in models:
        if isinstance(entry, str):
            normalized.append({"model": entry, "name": entry, "temperature": 0.2, "n": 1})
            continue
        if not isinstance(entry, dict):
            raise ValueError("Each model entry must be a string or an object.")
        model = entry.get("model") or entry.get("name")
        if not model:
            raise ValueError("Each model entry must define 'model' or 'name'.")
        normalized.append({
            "model": model,
            "name": entry.get("name", model),
            "temperature": entry.get("temperature", 0.2),
            "n": entry.get("n", 1),
        })
    return normalized


def _normalize_label_sets(label_sets: Dict, base_dir: str, defaults: dict) -> dict:
    normalized = {}
    for type_id, entry in label_sets.items():
        if isinstance(entry, str):
            normalized[type_id] = {
                "path": _resolve_path(base_dir, entry),
                "use_defaults": defaults["use_defaults"],
                "override_defaults": defaults["override_defaults"],
            }
            continue
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError("Each label set must be a string path or an object with a 'path' field.")
        normalized[type_id] = {
            "path": _resolve_path(base_dir, entry["path"]),
            "use_defaults": entry.get("use_defaults", defaults["use_defaults"]),
            "override_defaults": entry.get("override_defaults", defaults["override_defaults"]),
        }
    return normalized


def _compare_labels(predicted: list[int], expected: list[int], match_mode: str) -> tuple[bool, float]:
    if match_mode != "exact":
        raise ValueError(f"Unsupported match mode: {match_mode}")
    if len(predicted) != len(expected):
        return False, 0.0
    correct = sum(1 for p, e in zip(predicted, expected) if p == e)
    return predicted == expected, correct / len(expected) if expected else 0.0


def run_eval_from_config(
    config_path: str,
    api_key: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
    call_model_fn: Optional[Callable[..., list[str]]] = None,
) -> dict:
    config_path = os.path.abspath(config_path)
    base_dir = os.path.dirname(config_path)
    config = _load_json(config_path)

    dataset_path = _resolve_path(base_dir, config.get("dataset_path"))
    if not dataset_path:
        raise ValueError("Config must include dataset_path.")

    label_sets = config.get("label_sets")
    if not isinstance(label_sets, dict) or not label_sets:
        raise ValueError("Config must include a non-empty label_sets mapping.")

    prompt_sets = config.get("prompt_sets")
    if not prompt_sets:
        raise ValueError("Config must include prompt_sets.")

    models = config.get("models")
    if not models:
        raise ValueError("Config must include models.")

    output_dir = _resolve_path(base_dir, config.get("output_dir", "eval_runs"))
    os.makedirs(output_dir, exist_ok=True)

    defaults = {
        "use_defaults": config.get("use_defaults", True),
        "override_defaults": config.get("override_defaults", False),
    }
    match_mode = config.get("match_mode", "exact")

    prompt_sets_norm = _normalize_prompt_sets(prompt_sets, base_dir)
    models_norm = _normalize_models(models)
    label_sets_norm = _normalize_label_sets(label_sets, base_dir, defaults)

    api_key = api_key or os.getenv(api_key_env)
    if not api_key:
        raise ValueError("API key not provided. Use --api-key or set the environment variable.")

    dataset = _load_json(dataset_path)
    if not isinstance(dataset, dict):
        raise ValueError("Dataset must be a JSON object mapping IDs to examples.")

    caller = call_model_fn or call_model
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_suffix = uuid.uuid4().hex[:8]

    output_files = []

    for prompt_set in prompt_sets_norm:
        prompt_text = _load_text(prompt_set["path"])
        for model_cfg in models_norm:
            tag = f"{run_id}_{_slugify(prompt_set['name'])}_{_slugify(model_cfg['name'])}_t{model_cfg['temperature']}"
            tag = f"{tag}_{run_suffix}"
            results_path = os.path.join(output_dir, f"{tag}.jsonl")
            summary_path = os.path.join(output_dir, f"{tag}_summary.json")
            config_snapshot_path = os.path.join(output_dir, f"{tag}_config.json")

            type_stats: Dict[str, Dict[str, float]] = {}
            total = 0
            total_exact = 0
            total_segment_correct = 0
            total_segments = 0

            categories_cache: Dict[str, list[str]] = {}

            with open(results_path, "w", encoding="utf-8") as results_handle:
                for example_id, example in dataset.items():
                    type_id = example.get("typeId")
                    if type_id not in label_sets_norm:
                        raise ValueError(f"No label set configured for typeId '{type_id}'.")

                    if type_id not in categories_cache:
                        label_cfg = label_sets_norm[type_id]
                        categories_cache[type_id] = load_categories(
                            label_cfg["path"],
                            use_defaults=label_cfg["use_defaults"],
                            override=label_cfg["override_defaults"],
                            defaults=DEFAULT_CATEGORIES,
                        )

                    segments_map = example.get("segments", {})
                    if not isinstance(segments_map, dict):
                        raise ValueError(f"Example {example_id} has invalid segments.")
                    segments = list(segments_map.keys())
                    expected = list(segments_map.values())

                    prompt = build_prompt(segments, categories_cache[type_id], prompt_text)
                    raw_responses = caller(
                        prompt=prompt,
                        model=model_cfg["model"],
                        temperature=model_cfg["temperature"],
                        n=model_cfg["n"],
                        api_key=api_key,
                        system_prompt=prompt_set.get("system_prompt"),
                    )

                    predicted = []
                    error = None
                    for response_text in raw_responses:
                        try:
                            predicted = parse_labels(response_text)
                            break
                        except ValueError as exc:
                            error = str(exc)

                    if not predicted and error is None:
                        error = "No responses returned from model."

                    exact_match, segment_accuracy = _compare_labels(predicted, expected, match_mode)

                    total += 1
                    if exact_match:
                        total_exact += 1
                    total_segments += len(expected)
                    total_segment_correct += int(segment_accuracy * len(expected))

                    type_stat = type_stats.setdefault(type_id, {
                        "total": 0,
                        "exact": 0,
                        "segment_correct": 0,
                        "segments": 0,
                    })
                    type_stat["total"] += 1
                    type_stat["exact"] += 1 if exact_match else 0
                    type_stat["segment_correct"] += int(segment_accuracy * len(expected))
                    type_stat["segments"] += len(expected)

                    record = {
                        "example_id": example_id,
                        "type_id": type_id,
                        "text": example.get("text"),
                        "segments": segments,
                        "expected": expected,
                        "predicted": predicted,
                        "exact_match": exact_match,
                        "segment_accuracy": segment_accuracy,
                        "prompt_name": prompt_set["name"],
                        "prompt_file": prompt_set["path"],
                        "model": model_cfg["model"],
                        "temperature": model_cfg["temperature"],
                        "n": model_cfg["n"],
                        "raw_responses": raw_responses,
                        "error": error,
                    }
                    results_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            summary = {
                "run_id": tag,
                "dataset_path": dataset_path,
                "prompt_name": prompt_set["name"],
                "prompt_file": prompt_set["path"],
                "model": model_cfg["model"],
                "temperature": model_cfg["temperature"],
                "n": model_cfg["n"],
                "match_mode": match_mode,
                "total_examples": total,
                "exact_match_rate": total_exact / total if total else 0.0,
                "segment_accuracy": total_segment_correct / total_segments if total_segments else 0.0,
                "by_type": {},
            }
            for type_id, stats in type_stats.items():
                summary["by_type"][type_id] = {
                    "total_examples": stats["total"],
                    "exact_match_rate": stats["exact"] / stats["total"] if stats["total"] else 0.0,
                    "segment_accuracy": stats["segment_correct"] / stats["segments"] if stats["segments"] else 0.0,
                }

            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)

            config_snapshot = {
                "config_path": config_path,
                "dataset_path": dataset_path,
                "output_dir": output_dir,
                "label_sets": label_sets_norm,
                "prompt_set": prompt_set,
                "model": model_cfg,
                "defaults": defaults,
                "match_mode": match_mode,
            }
            with open(config_snapshot_path, "w", encoding="utf-8") as handle:
                json.dump(config_snapshot, handle, ensure_ascii=False, indent=2)

            output_files.append({
                "results": results_path,
                "summary": summary_path,
                "config": config_snapshot_path,
            })

    return {
        "output_dir": output_dir,
        "runs": output_files,
    }
