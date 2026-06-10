# phrase-labeler

**Tag parts of a sentence with the categories you care about, using OpenAI.**

You give `phrase-labeler` a sentence and a short list of labels (for example
`["Stakeholders", "Setting", "Goal", "Obstacle", "Constraints"]`). It returns
the pieces of the sentence that match each label, with exact character
positions. Pieces can overlap — the same word can belong to more than one
label — which is useful whenever a phrase plays multiple roles at once.

No fine-tuning, no training data. Just a prompt and a category list.

```text
Input  : "When editing visualizations, users face difficulty navigating GUIs."
Labels : ["Stakeholders", "Setting", "Goal", "Obstacle", "Constraints"]

Output : "When editing visualizations"  -> Goal
         "users"                        -> Stakeholders
         "face difficulty navigating"   -> Obstacle
         "GUIs"                         -> Setting
```

---

## Table of contents

1. [Install](#1-install)
2. [Quick start (Python)](#2-quick-start-python)
3. [Quick start (command line)](#3-quick-start-command-line)
4. [The two modes](#4-the-two-modes)
5. [Bringing your own categories](#5-bringing-your-own-categories)
6. [Teaching the model your corrections](#6-teaching-the-model-your-corrections-negative-examples)
7. [Customizing the prompt](#7-customizing-the-prompt-advanced)
8. [Troubleshooting](#8-troubleshooting)
9. [Developing on this repo](#9-developing-on-this-repo)

---

## 1. Install

**Requirements**: Python 3.10+ and an OpenAI API key.

```bash
pip install phrase-labeler
```

Get an API key at <https://platform.openai.com/api-keys>, then make it available
to the package. Either option works:

**Option A — export it in your shell** (simplest):

```bash
export OPENAI_API_KEY="sk-..."           # macOS / Linux
setx OPENAI_API_KEY "sk-..."             # Windows PowerShell (new sessions)
```

**Option B — pass it directly in Python**:

```python
find_labels_multi(..., api_key="sk-...")
```

Verify the install worked:

```bash
label-phrase --help
```

---

## 2. Quick start (Python)

This is the whole flow: one import, one call.

```python
import os
from phrase_labeler import find_labels_multi

spans = find_labels_multi(
    sentence="DynaVis blends natural language input with dynamically generated widgets.",
    api_key=os.environ["OPENAI_API_KEY"],
    categories=["Artifact", "Method", "Contribution", "Interaction"],
)

for s in spans:
    print(s)
```

Typical output:

```python
{'text': 'DynaVis', 'label': 0, 'start': 0, 'end': 7}
{'text': 'blends natural language input', 'label': 1, 'start': 8, 'end': 37}
{'text': 'with dynamically generated widgets', 'label': 0, 'start': 38, 'end': 72}
```

What you get back:

| Field | Meaning |
| ----- | ------- |
| `text` | The exact substring from your sentence. |
| `label` | Index into the `categories` list you passed (`0` = `"Artifact"` above). |
| `start` / `end` | Character offsets into the sentence. `sentence[start:end] == text`. |

A label of `-1` means "doesn't fit any category" (rhetorical scaffolding like
*"In this paper,"*).

---

## 3. Quick start (command line)

The package installs a `label-phrase` command:

```bash
label-phrase "DynaVis blends natural language input with dynamically generated widgets." \
    "$OPENAI_API_KEY" \
    --multi-label
```

Output is printed as a JSON array of spans.

---

## 4. The two modes

`phrase-labeler` has two APIs. Unless you know you need the legacy one, **use
multi-label**.

### Multi-label mode (recommended)

- **Input**: a raw, unmodified sentence.
- **Output**: a list of spans with `text`, `label`, `start`, `end`.
- **Overlapping labels allowed** — the same word can appear in multiple spans.
- **How to use it**: `find_labels_multi(...)` in Python, or the `--multi-label`
  flag on the CLI.

### Single-label mode (legacy)

- **Input**: a **pre-segmented** list of string chunks, e.g. `["When editing,", "users face difficulty."]`.
- **Output**: one label per chunk (no overlap).
- **How to use it**: `find_labels(...)` in Python, or omit `--multi-label` on the CLI.

Use the legacy mode only if you already have sentence segmentation from
somewhere else and want one label per pre-cut chunk. Everyone else:
multi-label.

---

## 5. Bringing your own categories

The categories can be anything — a handful of strings that describe what you
want to tag. Three ways to supply them:

### 5a. Inline list (simplest)

```python
find_labels_multi(
    sentence="...",
    api_key=...,
    categories=["Stakeholders", "Setting", "Goal", "Obstacle", "Constraints"],
)
```

### 5b. Inline list + descriptions

Descriptions are injected into the prompt so the model understands each
category better. Recommended whenever category names are ambiguous.

```python
find_labels_multi(
    sentence="...",
    api_key=...,
    categories=["Stakeholders", "Setting", "Goal", "Obstacle", "Constraints"],
    category_descriptions=[
        "The people or groups the work is about",
        "Where the problem takes place",
        "What users are trying to do",
        "Why the task is hard",
        "Fixed limits on the solution",
    ],
)
```

### 5c. JSON file (good for reuse and for the CLI)

Create a file, e.g. `my_categories.json`:

```json
{
  "description": "Short one-liner about this taxonomy; shown to the model.",
  "labels": {
    "0": "Stakeholders",
    "1": "Setting",
    "2": "Goal",
    "3": "Obstacle",
    "4": "Constraints"
  }
}
```

Rules: keys must be contiguous integers starting at `0`. `description` is
optional.

Use it:

```bash
label-phrase "..." "$OPENAI_API_KEY" --multi-label \
    --override-categories my_categories.json
```

Or in Python:

```python
from phrase_labeler import load_categories, find_labels_multi

categories, description = load_categories("my_categories.json")
find_labels_multi(sentence="...", api_key=..., categories=categories, description=description)
```

---

## 6. Teaching the model your corrections (negative examples)

When the model gets a label wrong, you can hand back a correction without
retraining anything. A negative example says *"in this sentence, the piece of
text `X` should not be labeled as `Y`"*. Corrections are added to the prompt
the next time you call the labeler.

```python
find_labels_multi(
    sentence="When editing visualizations users face difficulty navigating GUIs.",
    api_key=...,
    categories=["Stakeholders", "Setting", "Goal", "Obstacle", "Constraints"],
    negative_examples=[
        {
            "sentence": "When editing visualizations users face difficulty navigating GUIs.",
            "segment_text": "When editing visualizations",
            "do_not_label_as": "Setting",
        },
    ],
)
```

Each correction is a dict with three keys:

| Key | What it holds |
| --- | ------------- |
| `sentence` | The full original sentence where the mistake happened. |
| `segment_text` | The exact piece of text that was mislabeled. |
| `do_not_label_as` | The label name (not the number) to avoid for that piece. |

### From the command line

Put them in a JSON file, e.g. `my_corrections.json`:

```json
[
  {
    "sentence": "When editing visualizations users face difficulty navigating GUIs.",
    "segment_text": "When editing visualizations",
    "do_not_label_as": "Setting"
  }
]
```

Then:

```bash
label-phrase "..." "$OPENAI_API_KEY" --multi-label \
    --negative-examples my_corrections.json
```

### A few notes

- Matching is per-sentence: the same piece of text can legitimately carry a
  different label in a different sentence, so corrections only apply when the
  full sentence matches.
- The correction is written into the prompt as guidance — the model is asked
  to respect it, but it is not a hard constraint.
- Ship as many corrections as you like; they accumulate.

---

## 7. Customizing the prompt (advanced)

You almost never need to do this. A sensible default prompt template is
bundled with the package. But if you want to override it:

### CLI

```bash
label-phrase "..." "$OPENAI_API_KEY" --multi-label --prompt-file my_prompt.txt
```

### Python

```python
with open("my_prompt.txt") as f:
    template = f.read()

find_labels_multi(..., prompt_template=template)
```

### Placeholders your template can use

| Placeholder | What it expands to |
| ----------- | ------------------ |
| `${sentence}` | The input sentence (or JSON list in legacy mode). |
| `${categories}` | Numbered category list (includes descriptions when provided). |
| `${category_count}` | Number of categories. |
| `${description}` | Category-set description, or empty string. |
| `${negative_examples}` | Formatted corrections, or empty string. |

---

## 8. Troubleshooting

| Problem | What to do |
| ------- | ---------- |
| `OPENAI_API_KEY` is not set / empty | Export it in your shell, or pass `api_key=` directly in Python. |
| `AuthenticationError` or `Incorrect API key` | Your key is wrong or expired — generate a new one at <https://platform.openai.com/api-keys>. |
| `NotFoundError: The model ... does not exist` | Pass a model you have access to, e.g. `model="gpt-5-mini"` or `model="gpt-4o-mini"`. |
| Empty output / `No response returned from model` | Your prompt likely hit a filter, or the sentence is too short. Try a different model. |
| Labels come back in a different order than expected | The `label` field is the index into your `categories` list; order of returned spans follows the order the model emits, not the categories list. |
| The package installs but `label-phrase --help` is not found | The command is installed by pip into your venv's `bin/` (or `Scripts/` on Windows); activate the venv, or call `python -m phrase_labeler.cli --help`. |

---

## 9. Developing on this repo

This section is for contributors and researchers — not needed to use the
package.

### Clone and install in editable mode

```bash
git clone https://github.com/ZiweiGu/Typical-phrase-labeler.git
cd Typical-phrase-labeler
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e .
```

Changes to source files in `phrase_labeler/` take effect immediately.

### Run the tests

```bash
python -m unittest tests/test_phrase_labeler.py -q
```

The suite mocks the OpenAI client, so no network or API key is required.

### Build and publish

```bash
pip install build twine
python -m build                  # produces dist/*.whl and dist/*.tar.gz
twine upload dist/*              # publishes to PyPI
```

Bump `version` in `pyproject.toml` before each release.

### Repo layout

```text
phrase_labeler/          # the pip package — everything shipped to PyPI
  __init__.py            # public API re-exports
  cli.py                 # label-phrase command + find_labels{,_multi}
  pipeline.py            # OpenAI call + response caching
  prompting.py           # prompt templating + default prompts
  categories.py          # category JSON loading

categories/              # example taxonomies (need/approach/novelty/default)
examples/                # sample negative-examples JSON
prompts/                 # reference prompt templates
tests/                   # unit tests (not shipped to PyPI)
eval/                    # research evaluation harness (not shipped to PyPI)
```

### Research / evaluation harness

The `eval/` directory contains batch evaluation tooling with MLflow logging,
accuracy reports, and exports of systematically-wrong segments. It is **not**
part of the pip package. See [eval/README.md](eval/README.md) for details.

---

## License

MIT — see [LICENSE](LICENSE).
