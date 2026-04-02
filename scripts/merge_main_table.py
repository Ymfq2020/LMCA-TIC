"""Merge baseline references with LMCA-TIC metrics."""

from __future__ import annotations

import argparse

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from lmca_tic.experiments.runner import merge_reference_baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merge_reference_baselines(args.reference, args.metrics, args.output)


if __name__ == "__main__":
    main()
