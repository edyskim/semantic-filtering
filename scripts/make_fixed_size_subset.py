#!/usr/bin/env python3
"""Create a stratified fixed-size subset from a CSV manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--target_n", type=int, required=True)
    parser.add_argument("--class_col", default="class_name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if len(df) < args.target_n:
        raise ValueError(f"Cannot sample {args.target_n} rows from {len(df)} rows.")

    _, sampled = train_test_split(
        df,
        test_size=args.target_n / len(df),
        stratify=df[args.class_col],
        random_state=args.seed,
    )
    if len(sampled) > args.target_n:
        sampled = sampled.sample(n=args.target_n, random_state=args.seed)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    sampled.reset_index(drop=True).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
