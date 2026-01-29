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
pip install -e .[eval,mlflow]
```

Set `OPENAI_API_KEY` in your environment or `.env` at the repo root (used by the eval harness).

### Build the package (for distribution)
```bash
python -m pip install build
python -m build
```

## Command-Line Usage

After installation, you can use the label-phrase command to label sentence segments. The syntax is as follows:

```bash
label-phrase "[sentence segments as a JSON list]" "[your-openai-api-key]" [path-to-categories.json] [--use-defaults|--no-use-defaults] [--override-defaults] [--prompt-file path-to-prompt.txt]
```

If you omit the categories file, the default 0-8 categories are used. If you provide a categories file, it can be a JSON object that maps numeric keys to labels:

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

By default, custom categories are appended after the defaults. Use `--override-defaults` to replace the corresponding default labels for the numeric keys you supply, or `--no-use-defaults` to use only the provided labels.

Example behaviors with defaults `[A, B, C]` and a file `{0: X, 2: Z}`:
```
--use-defaults (default) -> [A, B, C, X, Z]
--no-use-defaults        -> [X, Z]   # requires contiguous keys from 0
--override-defaults      -> [X, B, Z]
```

## Evaluation Harness

Use the standalone harness to evaluate prompts/models against a labeled dataset.

Example:
```bash
python scripts/run_eval.py --config eval_config.example.json --api-key YOUR_KEY
```

The eval script also reads `.env` from the repo root, so you can set `OPENAI_API_KEY` there.

The harness writes JSONL results and a summary JSON into a timestamped folder under `eval_runs/` (filenames omit timestamps). The config file lets you vary prompts, models, temperatures, and label sets.

Progress uses `tqdm` if installed (defaults to on), concurrency defaults to 1, and rate-limit retries use built-in defaults. These are intentionally not part of the experiment config.

If you provide `--prompt-file`, it should be a text file that uses `${sentence}`, `${categories}`, and optionally `${category_count}` placeholders. See `prompts/default.txt` for the default template.

## Testing

```bash
python -m unittest -q
```

## Module Layout

- `phrase_labeler/cli.py`: CLI entry point and end-to-end labeling flow.
- `phrase_labeler/prompting.py`: Prompt templating utilities and default categories.
- `phrase_labeler/pipeline.py`: LLM plumbing, response extraction, and prompt pipeline helpers.
- `prompts/`: Prompt templates for testing different prompt variations.
- `categories/`: Category lists for testing different label sets.
