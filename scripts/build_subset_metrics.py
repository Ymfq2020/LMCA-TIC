"""Build inductive subset / N-Shot metric files in batch.

Iterates over predictions produced by ``evaluate`` for every model and
runs ``evaluate_prediction_subset`` for the requested subset/shot
combinations. Output files are placed next to each prediction file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from lmca_tic.experiments.runner import evaluate_prediction_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        required=True,
        help="Processed dataset directory (contains filtered_targets.json)",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to a *_predictions.jsonl file produced by evaluate",
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="History-shot ceilings to evaluate inductive subset under",
    )
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    args = parser.parse_args()

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    evaluate_prediction_subset(
        processed_dir=args.processed_dir,
        predictions_path=args.predictions,
        output_path=str(out_prefix.parent / f"{out_prefix.name}_inductive.json"),
        split=args.split,
        subset="inductive",
    )
    evaluate_prediction_subset(
        processed_dir=args.processed_dir,
        predictions_path=args.predictions,
        output_path=str(out_prefix.parent / f"{out_prefix.name}_transductive.json"),
        split=args.split,
        subset="transductive",
    )
    for shot in args.shots:
        evaluate_prediction_subset(
            processed_dir=args.processed_dir,
            predictions_path=args.predictions,
            output_path=str(out_prefix.parent / f"{out_prefix.name}_inductive_shot{shot}.json"),
            split=args.split,
            subset="inductive",
            history_shot=shot,
        )


if __name__ == "__main__":
    main()
