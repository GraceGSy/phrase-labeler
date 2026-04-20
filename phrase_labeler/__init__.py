"""phrase_labeler — multi-label classification of sentence fragments with OpenAI.

Public API:

    from phrase_labeler import (
        find_labels,
        find_labels_multi,
        load_categories,
        DEFAULT_CATEGORIES,
        DEFAULT_PROMPT_TEMPLATE,
        DEFAULT_MULTI_LABEL_PROMPT_TEMPLATE,
        build_prompt,
        build_multi_label_prompt,
        format_negative_examples,
    )
"""

from .categories import load_categories, parse_categories_payload
from .cli import find_labels, find_labels_multi
from .prompting import (
    DEFAULT_CATEGORIES,
    DEFAULT_MULTI_LABEL_PROMPT_TEMPLATE,
    DEFAULT_PROMPT_TEMPLATE,
    build_multi_label_prompt,
    build_prompt,
    format_categories,
    format_negative_examples,
)

__all__ = [
    "find_labels",
    "find_labels_multi",
    "load_categories",
    "parse_categories_payload",
    "DEFAULT_CATEGORIES",
    "DEFAULT_PROMPT_TEMPLATE",
    "DEFAULT_MULTI_LABEL_PROMPT_TEMPLATE",
    "build_prompt",
    "build_multi_label_prompt",
    "format_categories",
    "format_negative_examples",
]
