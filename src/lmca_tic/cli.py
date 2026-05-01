"""Command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from lmca_tic.config.loader import load_experiment_config
from lmca_tic.data.bie_builder import OfflineBIEBuilder
from lmca_tic.data.preprocess import LocalTKGPreprocessor
from lmca_tic.experiments.runner import (
    DEFAULT_SEEDS,
    SUITES,
    aggregate_with_reference,
    evaluate_history_chain_sensitivity,
    evaluate_noise_sensitivity,
    evaluate_prediction_subset,
    merge_reference_baselines,
    run_experiment_suite,
    run_seeded_suite,
    run_train_ratio_sensitivity,
    run_window_sensitivity,
)
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
    suite_parser.add_argument("--config", nargs="+")
    suite_parser.add_argument("--suite", choices=sorted(SUITES))
    suite_parser.add_argument("--seeds", nargs="+", type=int)
    suite_parser.add_argument("--output-root", default="outputs/experiments")
    suite_parser.add_argument("--smoke", action="store_true")

    window_parser = subparsers.add_parser("window-sensitivity")
    window_parser.add_argument("--config", default="configs/experiments/full_icews14.yaml")
    window_parser.add_argument("--windows", nargs="+", type=int, default=[3, 7, 14, 21, 30])
    window_parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    window_parser.add_argument("--output-root", default="outputs/experiments/window_sensitivity")
    window_parser.add_argument("--smoke", action="store_true")

    train_ratio_parser = subparsers.add_parser("train-ratio")
    train_ratio_parser.add_argument("--config", default="configs/experiments/full_icews14.yaml")
    train_ratio_parser.add_argument("--ratios", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    train_ratio_parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    train_ratio_parser.add_argument("--output-root", default="outputs/experiments/train_ratio")
    train_ratio_parser.add_argument("--raw-output-root", default="data/derived/train_ratio")
    train_ratio_parser.add_argument("--smoke", action="store_true")

    history_chain_parser = subparsers.add_parser("history-chain")
    history_chain_parser.add_argument("--config", default="configs/experiments/full_icews14.yaml")
    history_chain_parser.add_argument("--checkpoint-dir")
    history_chain_parser.add_argument("--lengths", nargs="+", type=int, default=[1, 3, 5, 7, 10])
    history_chain_parser.add_argument("--output-root", default="outputs/experiments/history_chain")
    history_chain_parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    history_chain_parser.add_argument("--checkpoint", default="best.pt")

    noise_parser = subparsers.add_parser("noise-sensitivity")
    noise_parser.add_argument("--config", default="configs/experiments/full_icews14.yaml")
    noise_parser.add_argument("--checkpoint-dir")
    noise_parser.add_argument("--rates", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    noise_parser.add_argument("--output-root", default="outputs/experiments/noise_sensitivity")
    noise_parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    noise_parser.add_argument("--checkpoint", default="best.pt")
    noise_parser.add_argument("--seed", type=int, default=42)

    eval_subset_parser = subparsers.add_parser("eval-subset")
    eval_subset_parser.add_argument("--processed-dir", required=True)
    eval_subset_parser.add_argument("--predictions", required=True)
    eval_subset_parser.add_argument("--output", required=True)
    eval_subset_parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    eval_subset_parser.add_argument("--subset", default="all", choices=["all", "inductive", "transductive"])
    eval_subset_parser.add_argument("--history-shot", type=int)

    aggregate_parser = subparsers.add_parser("aggregate-results")
    aggregate_parser.add_argument("--reference", required=True)
    aggregate_parser.add_argument("--summary", required=True)
    aggregate_parser.add_argument("--output", required=True)

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
        if args.seeds:
            if args.config:
                config_paths = args.config
            elif args.suite:
                config_paths = SUITES[args.suite]
            else:
                config_paths = SUITES["main"]
            run_seeded_suite(
                config_paths=config_paths,
                seeds=args.seeds,
                output_root=args.output_root,
                smoke=args.smoke,
            )
        else:
            if args.config:
                config_paths = args.config
            elif args.suite:
                config_paths = SUITES[args.suite]
            else:
                raise ValueError("run-suite requires --config or --suite unless --seeds uses the default main suite.")
            run_experiment_suite(config_paths)
        return
    if args.command == "window-sensitivity":
        run_window_sensitivity(
            base_config_path=args.config,
            windows=args.windows,
            seeds=args.seeds,
            output_root=args.output_root,
            smoke=args.smoke,
        )
        return
    if args.command == "train-ratio":
        run_train_ratio_sensitivity(
            base_config_path=args.config,
            ratios=args.ratios,
            seeds=args.seeds,
            output_root=args.output_root,
            raw_output_root=args.raw_output_root,
            smoke=args.smoke,
        )
        return
    if args.command == "history-chain":
        evaluate_history_chain_sensitivity(
            config_path=args.config,
            checkpoint_dir=args.checkpoint_dir,
            lengths=args.lengths,
            output_root=args.output_root,
            split=args.split,
            checkpoint_name=args.checkpoint,
        )
        return
    if args.command == "noise-sensitivity":
        evaluate_noise_sensitivity(
            config_path=args.config,
            checkpoint_dir=args.checkpoint_dir,
            rates=args.rates,
            output_root=args.output_root,
            split=args.split,
            checkpoint_name=args.checkpoint,
            seed=args.seed,
        )
        return
    if args.command == "eval-subset":
        evaluate_prediction_subset(
            processed_dir=args.processed_dir,
            predictions_path=args.predictions,
            output_path=args.output,
            split=args.split,
            subset=args.subset,
            history_shot=args.history_shot,
        )
        return
    if args.command == "aggregate-results":
        aggregate_with_reference(args.reference, args.summary, args.output)
        return
    if args.command == "merge-baselines":
        merge_reference_baselines(args.reference, args.metrics, args.output)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
