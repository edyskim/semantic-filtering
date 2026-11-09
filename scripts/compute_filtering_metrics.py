#!/usr/bin/env python3
"""Compute binary keep/remove filtering metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score


def is_remove(value: object) -> int:
    text = str(value).strip().lower()
    return int(text.startswith("remove") or text.startswith("rule") or text in {"1", "true", "removed"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--ref_csv", required=True)
    parser.add_argument("--id_col", default="img_path")
    parser.add_argument("--pred_col", default="vlm_decision")
    parser.add_argument("--ref_col", default="expert_decision")
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    pred = pd.read_csv(args.pred_csv)
    ref = pd.read_csv(args.ref_csv)
    merged = pred[[args.id_col, args.pred_col]].merge(ref[[args.id_col, args.ref_col]], on=args.id_col)

    y_pred = merged[args.pred_col].map(is_remove)
    y_ref = merged[args.ref_col].map(is_remove)

    out = pd.DataFrame(
        [
            {
                "n": len(merged),
                "accuracy": accuracy_score(y_ref, y_pred),
                "precision": precision_score(y_ref, y_pred, zero_division=0),
                "recall": recall_score(y_ref, y_pred, zero_division=0),
                "f1": f1_score(y_ref, y_pred, zero_division=0),
                "cohen_kappa": cohen_kappa_score(y_ref, y_pred),
            }
        ]
    )
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
