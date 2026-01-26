import argparse
import ast
import json
import os
from typing import Dict, Optional

import openai

from .pipeline import LLM, Phrase_TaggerPromptPipeline, extract_responses
from .prompting import DEFAULT_CATEGORIES, DEFAULT_PROMPT_TEMPLATE, build_prompt


TEMPERATURE = 0.2 #The temperature for ChatGPT calls


def _load_prompt_template(prompt_file: Optional[str]) -> str:
    """Load a prompt template from disk or return the default."""
    if not prompt_file:
        return DEFAULT_PROMPT_TEMPLATE
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"File not found: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as handle:
        return handle.read()


def _normalize_label_map(raw_labels: Dict) -> Dict[int, str]:
    """Normalize a label map with numeric keys into {int: str}."""
    if not isinstance(raw_labels, dict):
        raise ValueError("Labels must be a JSON object mapping numeric keys to strings.")
    normalized = {}
    for key, value in raw_labels.items():
        if isinstance(key, int):
            idx = key
        elif isinstance(key, str) and key.isdigit():
            idx = int(key)
        else:
            raise ValueError("Category label keys must be non-negative integers.")
        if idx < 0:
            raise ValueError("Category label keys must be non-negative integers.")
        if not isinstance(value, str):
            raise ValueError("Category labels must be strings.")
        normalized[idx] = value
    return normalized


def _labels_from_map(label_map: Dict[int, str], require_contiguous: bool) -> list[str]:
    """Return label values ordered by numeric key, with optional contiguous validation."""
    if not label_map:
        return []
    keys_sorted = sorted(label_map.keys())
    if require_contiguous:
        expected = list(range(len(keys_sorted)))
        if keys_sorted != expected:
            raise ValueError("Category label keys must be contiguous starting at 0.")
    return [label_map[idx] for idx in keys_sorted]


def _parse_categories_payload(payload) -> Dict[int, str]:
    """Parse category JSON into a label map."""
    if isinstance(payload, list):
        if not all(isinstance(c, str) for c in payload):
            raise ValueError("The categories list must contain strings only.")
        return {i: c for i, c in enumerate(payload)}

    if isinstance(payload, dict):
        if "labels" in payload:
            raw_labels = payload["labels"]
        else:
            raw_labels = payload
        return _normalize_label_map(raw_labels)

    raise ValueError("The categories file must contain a JSON list or an object mapping numeric keys to labels.")


def _merge_categories(defaults: list[str], label_map: Dict[int, str], use_defaults: bool, override: bool) -> list[str]:
    """Merge user categories with defaults based on flags."""
    if override:
        categories = list(defaults)
        for idx, label in label_map.items():
            if idx >= len(categories):
                raise ValueError("Override label index out of range for default categories.")
            categories[idx] = label
        return categories
    if use_defaults:
        return list(defaults) + _labels_from_map(label_map, require_contiguous=False)
    return _labels_from_map(label_map, require_contiguous=True)


def _load_categories(categories_file: Optional[str], use_defaults: bool, override: bool) -> list[str]:
    """Load categories from disk and merge with defaults based on flags."""
    if not categories_file:
        return DEFAULT_CATEGORIES
    if not os.path.exists(categories_file):
        raise FileNotFoundError(f"File not found: {categories_file}")
    with open(categories_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    label_map = _parse_categories_payload(payload)
    if override:
        use_defaults = True
    return _merge_categories(DEFAULT_CATEGORIES, label_map, use_defaults, override)


def find_labels(segmented_sent, k, categories, prompt_template=None):
    """Classify each segment and print a labeled list to stdout."""
    output = []
    openai.api_key = k
    if prompt_template is None:
        prompt_template = DEFAULT_PROMPT_TEMPLATE
    filled_prompt = build_prompt(segmented_sent, categories, prompt_template)
    phrase_tagger = Phrase_TaggerPromptPipeline(filled_prompt)
    tmp = []
    phrase_tagger.clear_cached_responses()
    for res in phrase_tagger.gen_responses({"sentence": str(segmented_sent)}, LLM.ChatGPT, n=1, temperature=TEMPERATURE):
        tmp.extend(extract_responses(res, llm=LLM.ChatGPT))
    color_list = ast.literal_eval(tmp[0])
    if len(color_list) == len(segmented_sent):
        for j, segment in enumerate(segmented_sent):
            output.append({'text': segment, 'label': color_list[j]})
    else:
        print(segmented_sent)
        print(color_list)
        for j, segment in enumerate(segmented_sent):
            output.append({'text': segment, 'label': 0})
    print(output)


def main():
    """Parse CLI args, load categories, and print labels for the input segments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", type=json.loads,
                        help="The sentence segments as a JSON list of strings (e.g. '[\"This paper\", \"proposes a method\"]')")
    parser.add_argument("api_key", help="Your OpenAI API key")
    parser.add_argument("categories_file", type=str, nargs="?",
                        help="Optional path to JSON file containing a category list")
    parser.add_argument("--use-defaults", action=argparse.BooleanOptionalAction, default=True,
                        help="Include default categories before your custom labels (default: true).")
    parser.add_argument("--override-defaults", action="store_true",
                        help="Replace default labels at the provided indices.")
    parser.add_argument("--prompt-file", type=str,
                        help="Optional path to a prompt template file")

    args = parser.parse_args()

    categories = _load_categories(
        args.categories_file,
        use_defaults=args.use_defaults,
        override=args.override_defaults,
    )

    prompt_template = _load_prompt_template(args.prompt_file)
    find_labels(args.sentence, args.api_key, categories, prompt_template)
