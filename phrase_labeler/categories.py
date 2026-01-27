import json
from typing import Dict, Optional


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


def parse_categories_payload(payload) -> Dict[int, str]:
    """Parse category JSON into a label map."""
    if isinstance(payload, list):
        if not all(isinstance(c, str) for c in payload):
            raise ValueError("The categories list must contain strings only.")
        return {i: c for i, c in enumerate(payload)}

    if isinstance(payload, dict):
        raw_labels = payload.get("labels", payload)
        return normalize_label_map(raw_labels)

    raise ValueError("The categories file must contain a JSON list or an object mapping numeric keys to labels.")


def merge_categories(
    defaults: list[str],
    label_map: Dict[int, str],
    use_defaults: bool,
    override: bool,
) -> list[str]:
    """Merge user categories with defaults based on flags."""
    if override:
        categories = list(defaults)
        for idx, label in label_map.items():
            if idx >= len(categories):
                raise ValueError("Override label index out of range for default categories.")
            categories[idx] = label
        return categories
    if use_defaults:
        return list(defaults) + labels_from_map(label_map, require_contiguous=False)
    return labels_from_map(label_map, require_contiguous=True)


def load_label_map(categories_file: str) -> Dict[int, str]:
    """Load a categories file and return a label map."""
    with open(categories_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return parse_categories_payload(payload)


def load_categories(
    categories_file: Optional[str],
    use_defaults: bool,
    override: bool,
    defaults: Optional[list[str]] = None,
) -> list[str]:
    """Load categories from disk and merge with defaults based on flags."""
    defaults = defaults or []
    if not categories_file:
        return list(defaults)
    label_map = load_label_map(categories_file)
    if override:
        use_defaults = True
    return merge_categories(defaults, label_map, use_defaults, override)
