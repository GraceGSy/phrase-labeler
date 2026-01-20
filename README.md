# Phrase Labeler

A Python package that labels sentence segments given predefined segment labels using OpenAI API

## Installation

To install this package, run:

```bash
pip install phrase-labeler
```

## Command-Line Usage

After installation, you can use the label-phrase command to label sentence segments. The syntax is as follows:

```bash
label-phrase "[sentence segments as a JSON list]" "[your-openai-api-key]" [path-to-categories.json] [--extend-categories] [--prompt-file path-to-prompt.txt]
```

If you omit the categories file, the default 0-8 categories are used. If you provide a categories file, it must be a JSON list of strings (see `categories/default.json` or `phrase_labeler/example.json`). Use `--extend-categories` to append your categories to the defaults instead of replacing them.

If you provide `--prompt-file`, it should be a text file that uses `${sentence}`, `${categories}`, and optionally `${category_count}` placeholders. See `prompts/default.txt` for the default template.

## Module Layout

- `phrase_labeler/cli.py`: CLI entry point and end-to-end labeling flow.
- `phrase_labeler/prompting.py`: Prompt templating utilities and default categories.
- `phrase_labeler/pipeline.py`: LLM plumbing, response extraction, and prompt pipeline helpers.
- `prompts/`: Prompt templates for testing different prompt variations.
- `categories/`: Category lists for testing different label sets.
