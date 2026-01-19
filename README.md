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
label-phrase "[sentent segments as a JSON list]" "[your-openai-api-key]" "[path to json file that contains a list of predefined labels]"
```