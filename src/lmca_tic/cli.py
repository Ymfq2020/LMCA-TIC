"""Command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from lmca_tic.config.loader import load_experiment_config
from lmca_tic.data.preprocess import LocalTKGPreprocessor
from lmca_tic.experiments.runner import merge_reference_baselines, run_experiment_suite
from lmca_tic.training.trainer import LMCATICTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="LMCA-TIC")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("--config", required=True)

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
