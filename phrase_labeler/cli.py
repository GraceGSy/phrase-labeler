import argparse
import ast
import json
import os
import re
from typing import Optional

import openai

from .categories import load_categories
from .pipeline import LLM, Phrase_TaggerPromptPipeline, extract_responses
from .prompting import (
    DEFAULT_CATEGORIES,
    DEFAULT_MULTI_LABEL_PROMPT_TEMPLATE,
    DEFAULT_PROMPT_TEMPLATE,
    build_batch_multi_label_prompt,
    build_multi_label_prompt,
    build_prompt,
)


TEMPERATURE = None  # Let the model use its own default; some models reject non-default values


def _load_prompt_template(prompt_file: Optional[str]) -> Optional[str]:
    """Load a prompt template from disk, or return None to signal the default."""
    if not prompt_file:
        return None
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"File not found: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as handle:
        return handle.read()


def _load_negative_examples(path: Optional[str]) -> list[dict]:
    """Load a JSON file of negative examples, or return an empty list."""
    if not path:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Negative examples file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(
            "Negative examples file must contain a JSON array of "
            '{"sentence", "segment_text", "do_not_label_as"} objects.'
        )
    return payload


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    # Remove first line (```json or ```) and last line (```)
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_multi_label_response(raw_text: str) -> list[dict]:
    """Parse the LLM response into a list of {text, context, label} dicts."""
    cleaned = _strip_code_fences(raw_text)

    # Try to extract JSON array from the response
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    spans = json.loads(cleaned)
    if not isinstance(spans, list):
        raise ValueError("Expected a JSON array of span objects.")

    result = []
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError(f"Expected span object, got {type(span)}")
        text = span.get("text", "")
        context = span.get("context", text)
        label = span.get("label", -1)
        if not isinstance(label, int):
            label = int(label)
        result.append({"text": text, "context": context, "label": label})
    return result


def _resolve_spans(sentence: str, spans: list[dict]) -> list[dict]:
    """Resolve span positions using context for disambiguation.

    For each span, finds the context substring in the sentence, then
    locates the text within that context to get the character offsets.
    Falls back to sequential matching if context isn't found.
    """
    resolved = []
    for span in spans:
        text = span["text"]
        context = span.get("context", text)
        label = span["label"]

        # Find context in the sentence
        ctx_start = sentence.find(context)
        if ctx_start != -1:
            # Find text within the context region
            text_offset = context.find(text)
            if text_offset != -1:
                start = ctx_start + text_offset
                end = start + len(text)
                resolved.append({
                    "text": text,
                    "label": label,
                    "start": start,
                    "end": end,
                })
                continue

        # Fallback: find text directly in sentence (sequential)
        start = sentence.find(text)
        if start != -1:
            resolved.append({
                "text": text,
                "label": label,
                "start": start,
                "end": start + len(text),
            })
        else:
            # Last resort: case-insensitive search
            idx = sentence.lower().find(text.lower())
            if idx != -1:
                resolved.append({
                    "text": sentence[idx:idx + len(text)],
                    "label": label,
                    "start": idx,
                    "end": idx + len(text),
                })

    return resolved


def find_labels(
    segmented_sent,
    k,
    categories,
    prompt_template=None,
    description="",
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    negative_examples: Optional[list[dict]] = None,
):
    """Classify each segment and return a labeled list (legacy single-label API).

    Parameters
    ----------
    model : str, optional
        OpenAI model name. Defaults to gpt-5-mini when not provided.
    temperature : float, optional
        Sampling temperature. Ignored when reasoning_effort is set.
    reasoning_effort : str, optional
        Reasoning effort level (low/medium/high/xhigh) for reasoning models.
    negative_examples : list of dict, optional
        User-supplied {"sentence", "segment_text", "do_not_label_as"} hints
        that are prepended to the prompt as calibration signal.
    """
    output = []
    openai.api_key = k
    if prompt_template is None:
        prompt_template = DEFAULT_PROMPT_TEMPLATE
    if temperature is None:
        temperature = TEMPERATURE
    filled_prompt = build_prompt(
        segmented_sent, categories, prompt_template, description,
        negative_examples=negative_examples,
    )
    phrase_tagger = Phrase_TaggerPromptPipeline(filled_prompt)
    tmp = []
    phrase_tagger.clear_cached_responses()
    for res in phrase_tagger.gen_responses(
        {"sentence": str(segmented_sent)},
        LLM.ChatGPT,
        n=1,
        temperature=temperature,
        model=model,
        reasoning_effort=reasoning_effort,
    ):
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
    return output


def find_labels_multi(
    sentence: str,
    api_key: str,
    categories: list[str],
    prompt_template: Optional[str] = None,
    description: str = "",
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    category_descriptions: list[str] | None = None,
    negative_examples: Optional[list[dict]] = None,
) -> list[dict]:
    """Classify a raw sentence into overlapping labeled spans (multi-label API).

    Unlike find_labels which takes pre-split segments, this takes the original
    sentence and returns overlapping spans with resolved character offsets.

    Parameters
    ----------
    prompt_template : str, optional
        Full prompt template string. If None, the packaged
        DEFAULT_MULTI_LABEL_PROMPT_TEMPLATE is used.
    negative_examples : list of dict, optional
        User-supplied {"sentence", "segment_text", "do_not_label_as"} hints
        prepended to the prompt. These instruct the model to avoid the
        specified label for the given sentence/segment pair.

    Returns
    -------
    list[dict]
        Each dict has: text (str), label (int), start (int), end (int).
    """
    openai.api_key = api_key
    if temperature is None:
        temperature = TEMPERATURE

    filled_prompt = build_multi_label_prompt(
        sentence, categories, prompt_template, description,
        category_descriptions=category_descriptions,
        negative_examples=negative_examples,
    )
    phrase_tagger = Phrase_TaggerPromptPipeline(filled_prompt)
    tmp = []
    phrase_tagger.clear_cached_responses()
    for res in phrase_tagger.gen_responses(
        {"sentence": sentence},
        LLM.ChatGPT,
        n=1,
        temperature=temperature,
        model=model,
        reasoning_effort=reasoning_effort,
    ):
        tmp.extend(extract_responses(res, llm=LLM.ChatGPT))

    if not tmp:
        raise ValueError("No response returned from model.")

    raw_spans = _parse_multi_label_response(tmp[0])
    resolved = _resolve_spans(sentence, raw_spans)
    return resolved


def _parse_batch_multi_label_response(raw_text: str) -> list[dict]:
    """Parse a batch LLM response into a list of span dicts with sentence_index."""
    cleaned = _strip_code_fences(raw_text)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    spans = json.loads(cleaned)
    if not isinstance(spans, list):
        raise ValueError("Expected a JSON array of span objects.")

    result = []
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError(f"Expected span object, got {type(span)}")
        sentence_index = span.get("sentence_index", 0)
        text = span.get("text", "")
        context = span.get("context", text)
        label = span.get("label", -1)
        if not isinstance(label, int):
            label = int(label)
        if not isinstance(sentence_index, int):
            sentence_index = int(sentence_index)
        result.append({
            "sentence_index": sentence_index,
            "text": text,
            "context": context,
            "label": label,
        })
    return result


def find_labels_multi_batch(
    sentences: list[str],
    api_key: str,
    categories: list[str],
    prompt_template: Optional[str] = None,
    description: str = "",
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    category_descriptions: list[str] | None = None,
    negative_examples: Optional[list[dict]] = None,
) -> list[list[dict]]:
    """Classify multiple sentences in a single OpenAI call.

    Sentences are embedded as XML tags in one prompt, so only one API call is
    made regardless of how many sentences there are. Results are returned as a
    list parallel to the input: result[i] contains the resolved spans for
    sentences[i].

    Returns
    -------
    list[list[dict]]
        One list of span dicts per input sentence.
        Each dict has: text (str), label (int), start (int), end (int).
    """
    if not sentences:
        return []

    openai.api_key = api_key
    if temperature is None:
        temperature = TEMPERATURE

    filled_prompt = build_batch_multi_label_prompt(
        sentences, categories, prompt_template, description,
        category_descriptions=category_descriptions,
        negative_examples=negative_examples,
    )

    # Phrase_TaggerPromptPipeline.gen_prompts substitutes ${sentence} from
    # properties; the batch template has no such placeholder, so the value is
    # unused — pass a dummy string to satisfy the dict access.
    phrase_tagger = Phrase_TaggerPromptPipeline(filled_prompt)
    tmp = []
    phrase_tagger.clear_cached_responses()
    for res in phrase_tagger.gen_responses(
        {"sentence": "__batch__"},
        LLM.ChatGPT,
        n=1,
        temperature=temperature,
        model=model,
        reasoning_effort=reasoning_effort,
    ):
        tmp.extend(extract_responses(res, llm=LLM.ChatGPT))

    if not tmp:
        raise ValueError("No response returned from model.")

    raw_spans = _parse_batch_multi_label_response(tmp[0])

    # Group spans by sentence_index
    groups: list[list[dict]] = [[] for _ in sentences]
    for span in raw_spans:
        idx = span["sentence_index"]
        if 0 <= idx < len(sentences):
            groups[idx].append(span)

    return [_resolve_spans(sentences[i], groups[i]) for i in range(len(sentences))]


def main():
    """Parse CLI args, load categories, and print labels for the input segments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", type=str,
                        help="The sentence text to label (or a JSON list of segments for legacy mode)")
    parser.add_argument("api_key", help="Your OpenAI API key")
    parser.add_argument("--override-categories", type=str, metavar="FILE",
                        help="Path to a JSON file with categories to use instead of the defaults")
    parser.add_argument("--prompt-file", type=str,
                        help="Optional path to a prompt template file")
    parser.add_argument("--multi-label", action="store_true",
                        help="Use multi-label overlapping span mode instead of legacy single-label")
    parser.add_argument("--negative-examples", type=str, metavar="FILE",
                        help="Path to a JSON file of user-marked negative examples "
                             "(list of {sentence, segment_text, do_not_label_as})")

    args = parser.parse_args()

    categories, description = load_categories(args.override_categories, defaults=DEFAULT_CATEGORIES)
    prompt_template = _load_prompt_template(args.prompt_file)
    negative_examples = _load_negative_examples(args.negative_examples)

    if args.multi_label:
        result = find_labels_multi(
            args.sentence, args.api_key, categories, prompt_template, description,
            negative_examples=negative_examples,
        )
        print(json.dumps(result, indent=2))
    else:
        segments = json.loads(args.sentence)
        # Pass the single-label default explicitly; None would leave the template at
        # the multi-label default which does not fit the single-label API surface.
        template = prompt_template if prompt_template is not None else DEFAULT_PROMPT_TEMPLATE
        find_labels(
            segments, args.api_key, categories, template, description,
            negative_examples=negative_examples,
        )
