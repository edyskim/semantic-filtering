#!/usr/bin/env python3
"""Compute inter-annotator agreement for binary keep/remove labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def is_remove(value: object) -> int:
    text = str(value).strip().lower()
    return int(text.startswith("remove") or text.startswith("rule") or text in {"1", "true", "removed"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--annotator1_col", default="annotator1")
    parser.add_argument("--annotator2_col", default="annotator2")
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    a1 = df[args.annotator1_col].map(is_remove)
    a2 = df[args.annotator2_col].map(is_remove)

    out = pd.DataFrame(
        [
            {
                "n": len(df),
                "agreement": float((a1 == a2).mean()),
                "cohen_kappa": cohen_kappa_score(a1, a2),
            }
        ]
    )
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
