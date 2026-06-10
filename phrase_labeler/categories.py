import json
from typing import Dict, Optional, Tuple


def normalize_label_map(raw_labels: Dict) -> Dict[int, str]:
    """Normalize a label map with numeric keys into {int: str}."""
    if not isinstance(raw_labels, dict):
        raise ValueError("Labels must be a JSON object mapping numeric keys to strings.")
    normalized: Dict[int, str] = {}
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


def labels_from_map(label_map: Dict[int, str], require_contiguous: bool) -> list[str]:
    """Return label values ordered by numeric key, with optional contiguous validation."""
    if not label_map:
        return []
    keys_sorted = sorted(label_map.keys())
    if require_contiguous:
        expected = list(range(len(keys_sorted)))
        if keys_sorted != expected:
            raise ValueError("Category label keys must be contiguous starting at 0.")
    return [label_map[idx] for idx in keys_sorted]


def parse_categories_payload(payload) -> Tuple[Dict[int, str], str]:
    """Parse category JSON into a (label_map, description) tuple.

    Supported formats:
      - List:  ["Cat A", "Cat B"]  — no description
      - Object with labels key:  {"description": "...", "labels": {"0": "Cat A"}}
      - Legacy object (no labels key):  {"0": "Cat A", "1": "Cat B"}  — no description
    """
    if isinstance(payload, list):
        if not all(isinstance(c, str) for c in payload):
            raise ValueError("The categories list must contain strings only.")
        return {i: c for i, c in enumerate(payload)}, ""

    if isinstance(payload, dict):
        if "labels" in payload:
            desc = payload.get("description", "")
            description = desc if isinstance(desc, str) else ""
            return normalize_label_map(payload["labels"]), description
        # Legacy format: entire object is the label map
        return normalize_label_map(payload), ""

    raise ValueError("The categories file must contain a JSON list or an object mapping numeric keys to labels.")


def load_label_map(categories_file: str) -> Tuple[Dict[int, str], str]:
    """Load a categories file and return (label_map, description)."""
    with open(categories_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return parse_categories_payload(payload)


def load_categories(
    categories_file: Optional[str],
    defaults: Optional[list[str]] = None,
) -> Tuple[list[str], str]:
    """Load categories from a file, or return defaults if no file is given.

    Returns (categories, description).  description is "" when not specified
    in the file or when falling back to defaults.

    Pass categories_file=None to use the hardcoded defaults.
    Pass a path to use only that file's categories (keys must be contiguous from 0).
    """
    defaults = defaults or []
    if not categories_file:
        return list(defaults), ""
    label_map, description = load_label_map(categories_file)
    return labels_from_map(label_map, require_contiguous=True), description
