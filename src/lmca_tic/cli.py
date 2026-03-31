"""Command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from lmca_tic.config.loader import load_experiment_config
from lmca_tic.data.bie_builder import OfflineBIEBuilder
from lmca_tic.data.preprocess import LocalTKGPreprocessor
from lmca_tic.experiments.runner import merge_reference_baselines, run_experiment_suite
from lmca_tic.training.trainer import LMCATICTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="LMCA-TIC")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("--config", required=True)

    build_bie_parser = subparsers.add_parser("build-bie")
    build_bie_parser.add_argument("--config")
    build_bie_parser.add_argument("--raw-dir")
    build_bie_parser.add_argument("--output-path")
    build_bie_parser.add_argument("--delimiter", default="\t")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--smoke", action="store_true")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--checkpoint", default="best.pt")
    eval_parser.add_argument("--split", default="test", choices=["train", "valid", "test"])

    suite_parser = subparsers.add_parser("run-suite")
    suite_parser.add_argument("--config", nargs="+", required=True)

    baseline_parser = subparsers.add_parser("merge-baselines")
    baseline_parser.add_argument("--reference", required=True)
    baseline_parser.add_argument("--metrics", required=True)
    baseline_parser.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "preprocess":
        config = load_experiment_config(args.config)
        LocalTKGPreprocessor(config).run()
        return
    if args.command == "build-bie":
        if args.config:
            config = load_experiment_config(args.config)
            raw_dir = Path(args.raw_dir) if args.raw_dir else Path(config.raw_dir)
            if args.output_path:
                output_path = Path(args.output_path)
            elif config.bie_path:
                output_path = Path(config.bie_path)
            else:
                output_path = raw_dir.parent / "bie" / "entity_metadata.jsonl"
            delimiter = config.delimiter
        else:
            if not args.raw_dir or not args.output_path:
                raise ValueError("build-bie requires either --config or both --raw-dir and --output-path.")
            raw_dir = Path(args.raw_dir)
            output_path = Path(args.output_path)
            delimiter = args.delimiter
        OfflineBIEBuilder(delimiter=delimiter).build_from_dir(raw_dir=raw_dir, output_path=output_path)
        return
    if args.command == "train":
        config = load_experiment_config(args.config)
        LocalTKGPreprocessor(config).run()
        trainer = LMCATICTrainer(config=config, smoke_mode=args.smoke)
        trainer.train()
        return
    if args.command == "evaluate":
        config = load_experiment_config(args.config)
        trainer = LMCATICTrainer(config=config)
        trainer.evaluate(split=args.split, checkpoint_name=args.checkpoint)
        return
    if args.command == "run-suite":
        run_experiment_suite(args.config)
        return
    if args.command == "merge-baselines":
        merge_reference_baselines(args.reference, args.metrics, args.output)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
