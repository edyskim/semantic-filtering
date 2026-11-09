"""Recompute the filter-level results of the article from the released metadata.

Usage: python scripts/reproduce_filter_tables.py
Prints Cohen's kappa of every configuration (without few-shot exemplar images), the median over the
configurations used in the summaries, and the kappa of the automatic comparators, and checks every VLM value against
summaries/filter_metrics.csv.
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
MAIN = ["zero_shot", "few_shots", "cot", "few_shots_fixed"]


def metrics(g):
    y = g.expert_decision.eq("remove")
    p = g.vlm_decision.eq("remove")
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    both = y & p
    has_rule = g.vlm_rule.str.startswith("rule").any()
    return pd.Series({"n": len(g), "removed": int(p.sum()),
                      "precision": tp / (tp + fp) if tp + fp else 0.0,
                      "recall": tp / (tp + fn) if tp + fn else 0.0,
                      "kappa": cohen_kappa_score(y, p),
                      "attribution_acc": (g.vlm_rule[both] == g.expert_rule[both]).mean() if both.any() and has_rule else float("nan")})


def main():
    m = pd.read_csv(ROOT / "manifests/split_manifest.csv")
    ex = set(zip(m.dataset[m.fewshot_exemplar_k.notna()], m.image_id[m.fewshot_exemplar_k.notna()]))
    d = pd.read_csv(ROOT / "metadata/vlm_filtering_metadata.csv")
    d = d[[(a, b) not in ex for a, b in zip(d.dataset, d.image_id)]]
    cols = ["expert_decision", "expert_rule", "vlm_decision", "vlm_rule"]
    r = d.groupby(["dataset", "model", "prompt", "in_summaries"])[cols].apply(metrics).reset_index()

    ref = pd.read_csv(ROOT / "summaries/filter_metrics.csv")
    ref = ref[["dataset", "model", "prompt"] + [c for c in ref.columns if c.startswith("exexemplar_")]]
    j = r.merge(ref, on=["dataset", "model", "prompt"])
    worst = max((j[c] - j[f"exexemplar_{c}"]).abs().max() for c in ["precision", "recall", "kappa"])
    print(f"checked {len(j)} configurations against summaries/filter_metrics.csv, largest difference {worst:.2e}")

    t = r[r.prompt.isin(MAIN)].pivot_table(index=["model"], columns=["dataset", "prompt"], values="kappa")
    print("\nCohen's kappa without exemplar images\n", t.round(2).to_string())
    for ds, g in r[r.prompt.isin(MAIN) & r.in_summaries].groupby("dataset"):
        print(f"\n{ds}: median kappa over {len(g)} configurations {g.kappa.median():.2f} "
              f"(range {g.kappa.min():.2f} to {g.kappa.max():.2f})")
    print("\nComparators and ablations\n", r[~r.prompt.isin(MAIN)].pivot_table(index=["model", "prompt"], columns="dataset", values="kappa").round(2).to_string())

    # automatic comparators (Table 5), scored on the same images
    c = pd.read_csv(ROOT / "metadata/comparator_decisions.csv")
    c = c[[(a, b) not in ex for a, b in zip(c.dataset, c.image_id)]]
    e = d[["dataset", "image_id", "expert_decision"]].drop_duplicates()
    c = c.merge(e, on=["dataset", "image_id"])
    k = pd.Series({key: cohen_kappa_score(g.expert_decision.eq("remove"), g.decision.eq("remove"))
                   for key, g in c.groupby(["method", "dataset"])})
    print("\nAutomatic comparators, Cohen's kappa without exemplar images\n", k.unstack().round(2).to_string())


if __name__ == "__main__":
    main()
