# phrase-labeler

Label sentence fragments with overlapping semantic categories using the OpenAI API.

Give the package a sentence and a small taxonomy; it returns a list of character-aligned
spans, each tagged with one of your categories. The same word can belong to multiple
overlapping spans, reflecting how a phrase can serve multiple roles at once.

- **Pluggable taxonomies** — ship your own JSON category files with optional descriptions.
- **Overlapping spans** — the multi-label mode identifies fragments and lets them overlap.
- **Negative examples** — feed user corrections (`{sentence, segment_text, do_not_label_as}`) into the prompt to steer the model away from known bad classifications.
- **Zero-config defaults** — a default multi-label prompt template is bundled with the
  package, so end-users do not need any extra files to get started.

## Install

```bash
pip install phrase-labeler
```

Set your OpenAI key:

```bash
export OPENAI_API_KEY="sk-..."   # or put OPENAI_API_KEY in a .env file
```

## Quick start (Python)

```python
from phrase_labeler import find_labels_multi

spans = find_labels_multi(
    sentence="DynaVis blends natural language input with dynamically generated widgets.",
    api_key="sk-...",
    categories=["Artifact", "Method", "Contribution", "Interaction"],
    model="gpt-4o-mini",
)

for s in spans:
    print(s)
# {'text': 'DynaVis', 'label': 0, 'start': 0, 'end': 7}
# {'text': 'blends natural language input', 'label': 1, 'start': 8, 'end': 37}
# ...
```

Each span has `text`, `label` (index into your `categories` list, or `-1` for unclassifiable), and character offsets `start` / `end` into the original sentence.

### With category descriptions and negative examples

```python
from phrase_labeler import find_labels_multi

spans = find_labels_multi(
    sentence="When editing visualizations users face difficulty in navigating complex GUIs.",
    api_key="sk-...",
    categories=["Stakeholders", "Setting", "Goal", "Obstacle", "Constraints"],
    category_descriptions=[
        "Who is affected",
        "Where it happens",
        "What users want to do",
        "Why it is hard",
        "Fixed limits that shape the solution space",
    ],
    negative_examples=[
        {
            "sentence": "When editing visualizations users face difficulty in navigating complex GUIs.",
            "segment_text": "When editing visualizations",
            "do_not_label_as": "Setting",
        },
    ],
)
```

Negative examples are rendered into the prompt as calibration hints the model is asked
to respect. They travel with the prompt itself — no training or fine-tuning.

## Command-line usage

```bash
# Multi-label (new — recommended)
label-phrase "The raw sentence goes here." "$OPENAI_API_KEY" --multi-label

# Multi-label with custom categories and negative examples
label-phrase "The raw sentence goes here." "$OPENAI_API_KEY" \
    --multi-label \
    --override-categories categories/need.json \
    --negative-examples examples/sample_negative_examples.json

# Legacy single-label (pre-segmented input as a JSON list)
label-phrase '["When editing,", "users face difficulty."]' "$OPENAI_API_KEY"
```

### Flags

| Flag | Description |
| ---- | ----------- |
| `--multi-label` | Use the overlapping-span API on a raw sentence. Omit for legacy pre-segmented mode. |
| `--override-categories FILE` | JSON file of categories to use in place of the defaults. |
| `--prompt-file FILE` | Custom prompt template. When omitted, the bundled default is used. |
| `--negative-examples FILE` | JSON array of `{sentence, segment_text, do_not_label_as}` corrections. |

### Categories file format

```json
{
  "description": "Optional: a one-line summary of the taxonomy, injected into the prompt.",
  "labels": {
    "0": "Stakeholders",
    "1": "Setting",
    "2": "Goal",
    "3": "Obstacle",
    "4": "Constraints"
  }
}
```

Keys must be contiguous integers starting at `0`. See [categories/](categories/) for
shipped examples (`need.json`, `approach.json`, `novelty.json`, `default.json`).

### Negative-examples file format

A JSON array of objects:

```json
[
  {
    "sentence": "When editing visualizations users face difficulty...",
    "segment_text": "When editing visualizations",
    "do_not_label_as": "Setting"
  }
]
```

See [examples/sample_negative_examples.json](examples/sample_negative_examples.json).

Matching is (sentence, segment) pair-specific — the same segment text can carry a
legitimately different label in a different sentence. The model also treats the examples
as calibration hints for similar segments elsewhere.

### Prompt template placeholders

If you supply `--prompt-file`, your template can use any of:

- `${sentence}` — the input sentence (or JSON list in legacy mode)
- `${categories}` — rendered as `"0 Label\n1 Label\n..."`, including descriptions when present
- `${category_count}` — number of categories
- `${description}` — category-set description, or empty string
- `${negative_examples}` — formatted corrections section, or empty string

## Testing

```bash
python -m unittest tests/test_phrase_labeler.py -q
```

The test suite mocks the OpenAI client, so no network or API key is required.

## Building and publishing

```bash
pip install build twine
python -m build
twine upload dist/*
```

Bump `version` in `pyproject.toml` before each release.

## Package layout

```text
phrase_labeler/          # pip package — the only thing shipped to PyPI
  __init__.py            # public re-exports
  cli.py                 # label-phrase CLI + find_labels{,_multi}
  pipeline.py            # LLM communication + response caching
  prompting.py           # Prompt templating, default categories, default prompts
  categories.py          # Category JSON loading

categories/              # Example category taxonomies
examples/                # Sample negative-examples JSON
prompts/                 # Reference prompt templates (optional overrides)
tests/                   # Core package tests (mocked OpenAI)
```

## Research / evaluation tools

The `eval/` directory contains batch evaluation tooling (MLflow, accuracy reports, problem-segment
exports) that is **not** shipped with the pip package. See [eval/README.md](eval/README.md).

```bash
pip install -e .
pip install tqdm
pip install mlflow  # optional

python eval/run_eval.py --config eval_config.json --api-key "$OPENAI_API_KEY"
python -m eval.analyze_eval_results eval_runs/my-run/
python -m eval.export_often_wrong_segments_csv eval_runs/my-run/
python eval/log_to_mlflow.py --run-dir eval_runs/my-run/ --experiment-name my-experiment
```

Copy `eval_config.example.json` → `eval_config.json` and fill in your dataset, label
sets, prompt files, and models.

## License

MIT — see [LICENSE](LICENSE).
