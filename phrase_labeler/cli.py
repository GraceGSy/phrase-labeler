import argparse
import ast
import json
import os
from typing import Optional

import openai

from .categories import load_categories
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

    categories = load_categories(
        args.categories_file,
        use_defaults=args.use_defaults,
        override=args.override_defaults,
        defaults=DEFAULT_CATEGORIES,
    )

    prompt_template = _load_prompt_template(args.prompt_file)
    find_labels(args.sentence, args.api_key, categories, prompt_template)
