# Phrase Labeler

A Python package that labels sentence segments given predefined segment labels using OpenAI API

## Install and Build

### From PyPI (end users)

```bash
pip install phrase-labeler
```

### For local development

Create a virtual environment and install in editable mode (so changes to the source are immediately reflected):

```bash
python -m venv venv
venv\Scripts\activate
pip install -e .
```

Set `OPENAI_API_KEY` in your environment or `.env` at the repo root.

### Build and publish the package

```bash
# Install build tools (one-time)
pip install build twine

# Build source distribution and wheel
python -m build

# Upload to PyPI
twine upload dist/*

# Or upload to TestPyPI first
twine upload --repository testpypi dist/*
```

The built artifacts appear in `dist/`. Increment `version` in `pyproject.toml` before each release.

## Command-Line Usage

After installation, you can use the `label-phrase` command to label sentence segments:

```bash
label-phrase "[sentence segments as a JSON list]" "[your-openai-api-key]" [--override-categories path-to-categories.json] [--prompt-file path-to-prompt.txt]
```

**Categories:**

- Omit `--override-categories` to use the built-in default categories (0–8).
- Pass `--override-categories <file>` to use only the categories defined in that file (defaults are ignored entirely).

A categories file is a JSON object with a `labels` map:

```json
{
  "labels": {
    "0": "stakeholders",
    "1": "setting",
    "2": "goal",
    "3": "obstacle",
    "4": "constraints"
  }
}
```

You can optionally include a top-level `description` field. Its value is injected into the prompt via the `${description}` placeholder, giving the model context about the category set:

```json
{
  "description": "These categories describe the rhetorical structure of HCI research abstracts. 'Goal' refers to the stated objective of the study, 'Obstacle' refers to challenges the authors faced, etc.",
  "labels": {
    "0": "stakeholders",
    "1": "setting",
    "2": "goal",
    "3": "obstacle",
    "4": "constraints"
  }
}
```

**Custom prompt templates:**

Pass `--prompt-file` with a `.txt` file using any of these placeholders:

- `${sentence}` — the input segments as a JSON list
- `${categories}` — the numbered category list
- `${category_count}` — total number of categories
- `${description}` — the category set description (empty string if not set)

## Testing

```bash
python -m unittest tests/test_phrase_labeler.py -q
```

## Package Layout

The `phrase_labeler/` package (what gets installed via pip) contains only the core labeling functionality:

- `phrase_labeler/cli.py` — CLI entry point (`label-phrase` command)
- `phrase_labeler/prompting.py` — prompt templating and default categories
- `phrase_labeler/pipeline.py` — LLM communication and response caching
- `phrase_labeler/categories.py` — category JSON loading

## Research / Evaluation Tools

The `eval/` directory contains research and evaluation tooling that is **not part of the pip package**. See [eval/README.md](eval/README.md) for full documentation.

Quick start:

```bash
pip install -e .
pip install tqdm          # required for eval
pip install mlflow        # optional, for MLflow logging

python eval/run_eval.py --config eval_config.json --api-key YOUR_KEY
python -m eval.analyze_eval_results eval_runs/my-run/
python -m eval.export_often_wrong_segments_csv eval_runs/my-run/
python eval/log_to_mlflow.py --run-dir eval_runs/my-run/ --experiment-name my-experiment
```

Copy `eval_config.example.json` → `eval_config.json` and fill in your dataset path, label sets, prompt files, and models.
