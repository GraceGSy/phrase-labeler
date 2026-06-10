import argparse
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EVAL_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.mlflow_logger import log_run_dir_to_mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Log eval run outputs to MLflow.")
    parser.add_argument("--run-dir", required=True, help="Path to a single eval run directory.")
    parser.add_argument("--experiment-name", help="MLflow experiment name (optional).")
    parser.add_argument("--tracking-uri", help="MLflow tracking URI (optional).")
    parser.add_argument("--run-name-prefix", help="Prefix for MLflow run names.")
    parser.add_argument("--no-per-example-metrics", action="store_true",
                        help="Disable logging per-example metrics.")

    args = parser.parse_args()

    run_ids = log_run_dir_to_mlflow(
        args.run_dir,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        run_name_prefix=args.run_name_prefix,
        per_example_metrics=not args.no_per_example_metrics,
    )

    print(f"Logged {len(run_ids)} runs to MLflow.")


if __name__ == "__main__":
    main()
