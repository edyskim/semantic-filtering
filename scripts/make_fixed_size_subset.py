#!/usr/bin/env python3
"""Draw a class-stratified subset of exactly target_n images from a filtered training pool.

This is the subsampling of the fixed-size setting of the article (Section 2.6). The article draws one subset per
training seed (0, 42, 84, 126, 168) and passes the same seed to train_downstream_classifier.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def exact_subsample(df: pd.DataFrame, target: int, seed: int, class_col: str = "class_name") -> pd.DataFrame:
    """Subsample to exactly target rows without duplicates, stratified by class."""
    if len(df) <= target:
        return df.copy()
    n_classes = df[class_col].nunique()
    if len(df) - target < n_classes:
        parts = []
        for _, group in df.groupby(class_col):
            k = max(1, round(len(group) / len(df) * target))
            parts.append(group.sample(n=min(k, len(group)), random_state=seed))
        sub = pd.concat(parts)
        if len(sub) < target:
            rest = df[~df.index.isin(set(sub.index.tolist()))]
            sub = pd.concat([sub, rest.sample(n=min(target - len(sub), len(rest)), random_state=seed)])
        if len(sub) > target:
            sub = sub.sample(n=target, random_state=seed)
    else:
        _, sub = train_test_split(df, test_size=target / len(df), stratify=df[class_col], random_state=seed)
        # stratified rounding can miss the target by one image, so the subset is adjusted to exactly target_n
        if len(sub) > target:
            sub = sub.sample(n=target, random_state=seed)
        elif len(sub) < target:
            rest = df[~df.index.isin(set(sub.index.tolist()))]
            sub = pd.concat([sub, rest.sample(n=target - len(sub), random_state=seed)])
    sub = sub.reset_index(drop=True)
    assert len(sub) == target and sub["img_path"].nunique() == target
    return sub


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
    sub = exact_subsample(df, args.target_n, args.seed, args.class_col)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
