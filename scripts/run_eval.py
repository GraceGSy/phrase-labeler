import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from phrase_labeler.eval_harness import run_eval_from_config


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = _strip_quotes(value.strip())
            if key and key not in os.environ:
                os.environ[key] = value


def main() -> None:
    _load_dotenv(os.path.join(REPO_ROOT, ".env"))

    parser = argparse.ArgumentParser(description="Run evaluation harness against a dataset.")
    parser.add_argument("--config", required=True, help="Path to eval config JSON.")
    parser.add_argument("--api-key", help="OpenAI API key (overrides env var).")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="Environment variable name for the API key.")

    args = parser.parse_args()

    result = run_eval_from_config(
        args.config,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
