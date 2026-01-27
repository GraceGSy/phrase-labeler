import argparse
import json

from phrase_labeler.eval_harness import run_eval_from_config


def main() -> None:
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
