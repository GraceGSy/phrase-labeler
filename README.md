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
label-phrase "[sentence segments as a JSON list]" "[your-openai-api-key]" [path-to-categories.json] [--extend-categories]
```

If you omit the categories file, the default 0-8 categories are used. If you provide a categories file, it must be a JSON list of strings (see `phrase_labeler/example.json`). Use `--extend-categories` to append your categories to the defaults instead of replacing them.
