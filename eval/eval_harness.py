import ast
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from string import Template
from typing import Callable, Dict, Iterable, Optional

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from phrase_labeler.categories import load_categories
from phrase_labeler.prompting import DEFAULT_CATEGORIES, build_prompt, format_categories, format_sentence

ALLOWED_REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}


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


def parse_correction_labels(raw_text: str, expected_length: int) -> list[int]:
    labels = parse_labels(raw_text)
    if len(labels) != expected_length:
        raise ValueError(
            f"Correction length mismatch (predicted={len(labels)}, expected={expected_length})."
        )
    return labels


def call_model(
    prompt: str,
    model: str,
    temperature: Optional[float],
    reasoning_effort: Optional[str],
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
    params = {
        "model": model,
        "messages": messages,
        "n": n,
    }
    if temperature is not None:
        params["temperature"] = temperature
    if reasoning_effort is not None:
        params["reasoning_effort"] = reasoning_effort
    try:
        response = client.chat.completions.create(**params)
    except TypeError as exc:
        message = str(exc)
        if reasoning_effort is not None and "unexpected keyword argument 'reasoning_effort'" in message:
            raise TypeError(
                "This OpenAI SDK does not support reasoning_effort for chat.completions. "
                "Upgrade the openai package or omit reasoning_effort in config."
            ) from exc
        raise
    return [choice.message.content for choice in response.choices]


def _slugify(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum() or char in ("-", "_"):
            cleaned.append(char)
        elif char.isspace():
            cleaned.append("-")
    return "".join(cleaned) or "run"


def _unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    counter = 1
    while True:
        candidate = f"{base}_{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def _unique_dir(path: str) -> str:
    if not os.path.exists(path):
        return path
    counter = 1
    while True:
        candidate = f"{path}_{counter}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


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


def _normalize_reasoning_effort(entry: dict) -> Optional[str]:
    if "reasoning" in entry:
        raise ValueError(
            "Model config field 'reasoning' is not supported. Use 'reasoning_effort' instead."
        )

    effort = entry.get("reasoning_effort")

    if effort is None:
        return None
    if not isinstance(effort, str):
        raise ValueError(
            "Model reasoning_effort must be null or one of: low, medium, high, xhigh."
        )
    effort = effort.strip().lower()
    if effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError(
            "Model reasoning_effort must be null or one of: low, medium, high, xhigh."
        )
    return effort


def _normalize_models(models: Iterable) -> list[dict]:
    normalized = []
    for entry in models:
        if isinstance(entry, str):
            normalized.append({
                "model": entry,
                "name": entry,
                "temperature": None,
                "reasoning_effort": None,
                "n": 1,
            })
            continue
        if not isinstance(entry, dict):
            raise ValueError("Each model entry must be a string or an object.")
        model = entry.get("model") or entry.get("name")
        if not model:
            raise ValueError("Each model entry must define 'model' or 'name'.")
        temperature = entry.get("temperature")
        if temperature is not None:
            temperature = float(temperature)
        reasoning_effort = _normalize_reasoning_effort(entry)
        normalized.append({
            "model": model,
            "name": entry.get("name", model),
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "n": entry.get("n", 1),
        })
    return normalized


def _normalize_judge_config(judge: Optional[dict], base_dir: str) -> dict:
    if judge is None:
        return {
            "enabled": False,
            "mode": "correct_labels",
            "prompt_path": None,
            "system_prompt": None,
            "fallback_to_base_on_error": True,
            "model": None,
        }
    if not isinstance(judge, dict):
        raise ValueError("Config field 'judge' must be an object when provided.")

    enabled = bool(judge.get("enabled", False))
    mode = str(judge.get("mode", "correct_labels")).strip().lower()
    if mode != "correct_labels":
        raise ValueError("Judge mode must be 'correct_labels'.")

    normalized = {
        "enabled": enabled,
        "mode": mode,
        "prompt_path": None,
        "system_prompt": judge.get("system_prompt"),
        "fallback_to_base_on_error": bool(judge.get("fallback_to_base_on_error", True)),
        "model": None,
    }
    if not enabled:
        return normalized

    prompt_path = judge.get("prompt_path")
    if not isinstance(prompt_path, str) or not prompt_path.strip():
        raise ValueError("Judge config must include a non-empty 'prompt_path' when enabled.")
    normalized["prompt_path"] = _resolve_path(base_dir, prompt_path)

    if "model" not in judge:
        raise ValueError("Judge config must include a 'model' entry when enabled.")
    normalized["model"] = _normalize_models([judge["model"]])[0]
    return normalized


def _normalize_label_sets(label_sets: Dict, base_dir: str) -> dict:
    normalized = {}
    for type_id, entry in label_sets.items():
        if isinstance(entry, str):
            normalized[type_id] = {"path": _resolve_path(base_dir, entry)}
            continue
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError("Each label set must be a string path or an object with a 'path' field.")
        normalized[type_id] = {"path": _resolve_path(base_dir, entry["path"])}
    return normalized


def _build_judge_prompt(
    prompt_template: str,
    text: Optional[str],
    segments: list[str],
    categories: list[str],
    predicted: list[int],
    description: str = "",
) -> str:
    return Template(prompt_template).safe_substitute({
        "text": "" if text is None else str(text),
        "sentence": format_sentence(segments),
        "segments": format_sentence(segments),
        "categories": format_categories(categories),
        "category_count": len(categories),
        "category_max": len(categories) - 1 if categories else -1,
        "predicted": json.dumps(predicted, ensure_ascii=False),
        "predicted_pretty": json.dumps(predicted, ensure_ascii=False, indent=2),
        "description": description,
    })


def _compare_labels(predicted: list[int], expected: list[int], match_mode: str) -> tuple[bool, float]:
    if match_mode != "exact":
        raise ValueError(f"Unsupported match mode: {match_mode}")
    if len(predicted) != len(expected):
        return False, 0.0
    correct = sum(1 for p, e in zip(predicted, expected) if p == e)
    return predicted == expected, correct / len(expected) if expected else 0.0


def _error_code_from_exception(exc: Exception) -> Optional[str]:
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        return code
    error = getattr(exc, "error", None)
    if isinstance(error, dict):
        return error.get("code")
    message = str(exc)
    if "rate_limit_exceeded" in message:
        return "rate_limit_exceeded"
    if "insufficient_quota" in message:
        return "insufficient_quota"
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    if exc.__class__.__name__ == "RateLimitError":
        return True
    return _error_code_from_exception(exc) == "rate_limit_exceeded"


def _is_quota_error(exc: Exception) -> bool:
    return _error_code_from_exception(exc) == "insufficient_quota"


def _sleep_with_backoff(attempt: int, base_delay: float, max_delay: float, jitter: float) -> None:
    delay = min(max_delay, base_delay * (2 ** attempt))
    if jitter > 0:
        delay += random.random() * jitter
    time.sleep(delay)


def run_eval_from_config(
    config_path: str,
    api_key: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
    call_model_fn: Optional[Callable[..., list[str]]] = None,
    show_progress: Optional[bool] = None,
    judge_enabled: Optional[bool] = None,
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

    run_name = config.get("run_name")
    if not run_name or not isinstance(run_name, str):
        raise ValueError("Config must include a non-empty 'run_name' string.")

    match_mode = config.get("match_mode", "exact")
    retry_config = {
        "max_retries": 3,
        "base_delay": 0.5,
        "max_delay": 5.0,
        "jitter": 0.2,
    }
    concurrency = 1
    if show_progress is None:
        show_progress = True

    prompt_sets_norm = _normalize_prompt_sets(prompt_sets, base_dir)
    models_norm = _normalize_models(models)
    label_sets_norm = _normalize_label_sets(label_sets, base_dir)
    judge_config_input = config.get("judge")
    if judge_enabled is not None:
        if judge_config_input is None:
            judge_config_input = {}
        if not isinstance(judge_config_input, dict):
            raise ValueError("Config field 'judge' must be an object when provided.")
        judge_config_input = dict(judge_config_input)
        judge_config_input["enabled"] = judge_enabled
    judge_cfg = _normalize_judge_config(judge_config_input, base_dir)

    api_key = api_key or os.getenv(api_key_env)
    if not api_key:
        raise ValueError("API key not provided. Use --api-key or set the environment variable.")

    dataset = _load_json(dataset_path)
    if not isinstance(dataset, dict):
        raise ValueError("Dataset must be a JSON object mapping IDs to examples.")

    caller = call_model_fn or call_model
    run_dir = _unique_dir(os.path.join(output_dir, run_name))
    os.makedirs(run_dir, exist_ok=True)

    output_files = []

    categories_by_type: Dict[str, list[str]] = {}
    descriptions_by_type: Dict[str, str] = {}
    for type_id, label_cfg in label_sets_norm.items():
        cats, desc = load_categories(label_cfg["path"], defaults=DEFAULT_CATEGORIES)
        categories_by_type[type_id] = cats
        descriptions_by_type[type_id] = desc

    judge_prompt_template = None
    if judge_cfg["enabled"]:
        judge_prompt_template = _load_text(judge_cfg["prompt_path"])

    def call_with_retry(prompt_text: str, model_cfg: dict, system_prompt: Optional[str]) -> tuple[list[str], Optional[str]]:
        attempt = 0
        while True:
            try:
                responses = caller(
                    prompt=prompt_text,
                    model=model_cfg["model"],
                    temperature=model_cfg["temperature"],
                    reasoning_effort=model_cfg["reasoning_effort"],
                    n=model_cfg["n"],
                    api_key=api_key,
                    system_prompt=system_prompt,
                )
                return responses, None
            except Exception as exc:
                error = str(exc)
                if _is_quota_error(exc):
                    return [], error
                if _is_rate_limit_error(exc) and attempt < retry_config["max_retries"]:
                    _sleep_with_backoff(
                        attempt,
                        retry_config["base_delay"],
                        retry_config["max_delay"],
                        retry_config["jitter"],
                    )
                    attempt += 1
                    continue
                return [], error

    for prompt_set in prompt_sets_norm:
        prompt_text = _load_text(prompt_set["path"])
        for model_cfg in models_norm:
            reasoning_suffix = ""
            if model_cfg["reasoning_effort"] is not None:
                reasoning_suffix = f"_r{_slugify(model_cfg['reasoning_effort'])}"
            judge_suffix = ""
            if judge_cfg["enabled"]:
                judge_model_name = judge_cfg["model"]["name"]
                judge_suffix = f"_j-{_slugify(judge_cfg['mode'])}-{_slugify(judge_model_name)}"
            base_name = (
                f"{_slugify(prompt_set['name'])}_{_slugify(model_cfg['name'])}"
                f"_t{model_cfg['temperature']}{reasoning_suffix}{judge_suffix}"
            )
            results_path = _unique_path(os.path.join(run_dir, f"{base_name}.jsonl"))
            summary_path = _unique_path(os.path.join(run_dir, f"{base_name}_summary.json"))
            config_snapshot_path = _unique_path(os.path.join(run_dir, f"{base_name}_config.json"))

            def evaluate_example(example_id: str, example: dict) -> dict:
                type_id = example.get("typeId")
                if type_id not in categories_by_type:
                    raise ValueError(f"No label set configured for typeId '{type_id}'.")

                segments_map = example.get("segments", {})
                if not isinstance(segments_map, dict):
                    raise ValueError(f"Example {example_id} has invalid segments.")
                segments = list(segments_map.keys())
                expected = list(segments_map.values())

                prompt = build_prompt(segments, categories_by_type[type_id], prompt_text, descriptions_by_type[type_id])
                raw_responses = []
                predicted: list[int] = []
                final_predicted: list[int] = []
                judge_corrected: list[int] = []
                judge_raw_responses = []
                judge_error = None
                error = None

                raw_responses, error = call_with_retry(
                    prompt_text=prompt,
                    model_cfg=model_cfg,
                    system_prompt=prompt_set.get("system_prompt"),
                )
                for response_text in raw_responses:
                    try:
                        predicted = parse_labels(response_text)
                        error = None
                        break
                    except ValueError as exc:
                        error = str(exc)
                if not predicted and error is None:
                    error = "No responses returned from model."

                final_predicted = predicted
                judge_model_cfg = judge_cfg["model"] if judge_cfg["enabled"] else None
                if judge_cfg["enabled"]:
                    if error is None and predicted:
                        judge_prompt = _build_judge_prompt(
                            prompt_template=judge_prompt_template,
                            text=example.get("text"),
                            segments=segments,
                            categories=categories_by_type[type_id],
                            predicted=predicted,
                            description=descriptions_by_type[type_id],
                        )
                        judge_raw_responses, judge_error = call_with_retry(
                            prompt_text=judge_prompt,
                            model_cfg=judge_model_cfg,
                            system_prompt=judge_cfg.get("system_prompt"),
                        )
                        for response_text in judge_raw_responses:
                            try:
                                judge_corrected = parse_correction_labels(response_text, len(segments))
                                judge_error = None
                                break
                            except ValueError as exc:
                                judge_error = str(exc)
                        if judge_corrected:
                            final_predicted = judge_corrected
                        elif not judge_cfg["fallback_to_base_on_error"]:
                            final_predicted = []
                            error = f"Judge correction failed: {judge_error or 'No responses returned from judge.'}"
                    else:
                        judge_error = "Skipped judge correction because base prediction failed."

                exact_match, segment_accuracy = _compare_labels(final_predicted, expected, match_mode)

                return {
                    "example_id": example_id,
                    "type_id": type_id,
                    "text": example.get("text"),
                    "segments": segments,
                    "expected": expected,
                    "predicted": predicted,
                    "judge_corrected": judge_corrected,
                    "final_predicted": final_predicted,
                    "exact_match": exact_match,
                    "segment_accuracy": segment_accuracy,
                    "prompt_name": prompt_set["name"],
                    "prompt_file": prompt_set["path"],
                    "model": model_cfg["model"],
                    "temperature": model_cfg["temperature"],
                    "reasoning_effort": model_cfg["reasoning_effort"],
                    "n": model_cfg["n"],
                    "raw_responses": raw_responses,
                    "judge_enabled": judge_cfg["enabled"],
                    "judge_mode": judge_cfg["mode"] if judge_cfg["enabled"] else None,
                    "judge_model": judge_model_cfg["model"] if judge_model_cfg else None,
                    "judge_model_name": judge_model_cfg["name"] if judge_model_cfg else None,
                    "judge_temperature": judge_model_cfg["temperature"] if judge_model_cfg else None,
                    "judge_reasoning_effort": judge_model_cfg["reasoning_effort"] if judge_model_cfg else None,
                    "judge_n": judge_model_cfg["n"] if judge_model_cfg else None,
                    "judge_prompt_path": judge_cfg["prompt_path"] if judge_cfg["enabled"] else None,
                    "judge_raw_responses": judge_raw_responses,
                    "judge_error": judge_error,
                    "error": error,
                }

            records: list[dict] = []
            total_items = len(dataset)
            desc = f"{prompt_set['name']} / {model_cfg['name']}"

            if show_progress and tqdm:
                progress = tqdm(total=total_items, desc=desc)
            else:
                progress = None

            if concurrency == 1:
                for example_id, example in dataset.items():
                    records.append(evaluate_example(example_id, example))
                    if progress:
                        progress.update(1)
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = {
                        executor.submit(evaluate_example, example_id, example): example_id
                        for example_id, example in dataset.items()
                    }
                    for future in as_completed(futures):
                        records.append(future.result())
                        if progress:
                            progress.update(1)

            if progress:
                progress.close()

            type_stats: Dict[str, Dict[str, float]] = {}
            total = 0
            total_exact = 0
            total_segment_correct = 0
            total_segments = 0
            judge_attempted_examples = 0
            judge_corrected_examples = 0
            judge_fallback_examples = 0

            with open(results_path, "w", encoding="utf-8") as results_handle:
                for record in records:
                    total += 1
                    if record["exact_match"]:
                        total_exact += 1
                    total_segments += len(record["expected"])
                    total_segment_correct += int(record["segment_accuracy"] * len(record["expected"]))
                    if record.get("judge_enabled"):
                        judge_attempted_examples += 1
                        if record.get("judge_corrected"):
                            judge_corrected_examples += 1
                        elif record.get("judge_error"):
                            judge_fallback_examples += 1

                    type_stat = type_stats.setdefault(record["type_id"], {
                        "total": 0,
                        "exact": 0,
                        "segment_correct": 0,
                        "segments": 0,
                    })
                    type_stat["total"] += 1
                    type_stat["exact"] += 1 if record["exact_match"] else 0
                    type_stat["segment_correct"] += int(record["segment_accuracy"] * len(record["expected"]))
                    type_stat["segments"] += len(record["expected"])

                    results_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            summary = {
                "run_id": os.path.basename(run_dir),
                "dataset_path": dataset_path,
                "prompt_name": prompt_set["name"],
                "prompt_file": prompt_set["path"],
                "model": model_cfg["model"],
                "temperature": model_cfg["temperature"],
                "reasoning_effort": model_cfg["reasoning_effort"],
                "n": model_cfg["n"],
                "match_mode": match_mode,
                "total_examples": total,
                "exact_match_rate": total_exact / total if total else 0.0,
                "segment_accuracy": total_segment_correct / total_segments if total_segments else 0.0,
                "uses_final_predicted": bool(judge_cfg["enabled"]),
                "judge": {
                    "enabled": judge_cfg["enabled"],
                    "mode": judge_cfg["mode"] if judge_cfg["enabled"] else None,
                    "prompt_path": judge_cfg["prompt_path"] if judge_cfg["enabled"] else None,
                    "fallback_to_base_on_error": judge_cfg["fallback_to_base_on_error"],
                    "model": judge_cfg["model"] if judge_cfg["enabled"] else None,
                    "examples_with_judge_enabled": judge_attempted_examples,
                    "examples_judge_corrected": judge_corrected_examples,
                    "examples_judge_fallback": judge_fallback_examples,
                },
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
                "run_dir": run_dir,
                "label_sets": label_sets_norm,
                "prompt_set": prompt_set,
                "model": model_cfg,
                "judge": judge_cfg,
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
        "run_dir": run_dir,
        "runs": output_files,
    }
