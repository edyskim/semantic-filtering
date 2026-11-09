"""Recompute the rule-level analysis of the article from the released decisions.

For every VLM configuration, rule-free and self-consistency run, and automatic comparator, the script computes the share of
the images removed by the experts under each rule that the filter also removed, and the share of the images kept by the
experts that the filter removed, on the training pools without the few-shot exemplar images. It checks every value against
summaries/rule_detection.csv, prints the summary rows of the main-text table, and counts the images on which most of the
17 valid VLM configurations disagree with the experts.

usage: python scripts/reproduce_rule_detection.py            (run from the repository root)
"""
import numpy as np
import pandas as pd

RULES = ["rule1", "rule2", "rule3", "rule4", "rule5"]
VALID_PROMPTS = {"zero_shot", "few_shots", "cot", "few_shots_fixed"}
SELECTED = {
    "plantdoc": ("gemini-2.0-flash-lite", "zero_shot"),
    "plantwild": ("claude-sonnet-4-6", "cot"),
}


def detection(removed, rule):
    r = {f"det_{k}": removed[rule == k].mean() for k in RULES}
    r["valid_removed"] = removed[rule == "none"].mean()
    return r


def main():
    meta = pd.read_csv("metadata/vlm_filtering_metadata.csv")
    comp = pd.read_csv("metadata/comparator_decisions.csv")
    man = pd.read_csv("manifests/split_manifest.csv")
    ref = pd.read_csv("summaries/rule_detection.csv", keep_default_na=False)
    cols = [f"det_{k}" for k in RULES] + ["valid_removed"]
    rows = []
    for ds in ["plantdoc", "plantwild"]:
        pool = man[(man.dataset == ds) & (man.set == "training_pool") & man.fewshot_exemplar_k.isna()].set_index("image_id")
        rule = pool.expert_rule
        m = meta[(meta.dataset == ds) & meta.in_summaries & meta.image_id.isin(pool.index)]
        for (model, prompt), g in m.groupby(["model", "prompt"]):
            if prompt not in VALID_PROMPTS | {"rule_free", "sc5_cot"}:
                continue
            removed = (g.set_index("image_id").vlm_decision.reindex(pool.index) == "remove")
            kind = {"rule_free": "rule_free", "sc5_cot": "self_consistency"}.get(prompt, "vlm")
            rows.append(dict(dataset=ds, kind=kind, model=model, prompt=prompt, **detection(removed, rule)))
        for method, g in comp[comp.dataset == ds].groupby("method"):
            removed = (g.set_index("image_id").decision.reindex(pool.index) == "remove")
            rows.append(dict(dataset=ds, kind="comparator", model=method, prompt="", **detection(removed, rule)))
    out = pd.DataFrame(rows)

    chk = out.merge(ref, on=["dataset", "kind", "model", "prompt"], suffixes=("", "_ref"))
    assert len(chk) == len(ref) == len(out), (len(chk), len(ref), len(out))
    dev = max((chk[c] - chk[f"{c}_ref"]).abs().max() for c in cols)
    print(f"checked {len(chk)} filters against summaries/rule_detection.csv, largest difference {dev:.2e}")

    pct = lambda x: f"{100 * x:4.0f}"
    print("\nDetection by rule (%), R1 to R5, and share of expert-kept images removed (Valid)")
    for ds in ["plantdoc", "plantwild"]:
        g = out[out.dataset == ds]
        print(f"\n{ds}")
        lines = [("VLM, median of 17 configurations", g[g.kind == "vlm"][cols].median()),
                 ("VLM, selected by kappa", g[(g.model == SELECTED[ds][0]) & (g.prompt == SELECTED[ds][1])][cols].iloc[0]),
                 ("Rule-free prompt, median of 4 models", g[g.kind == "rule_free"][cols].median())]
        lines += [(r.model, r[cols]) for _, r in g[g.kind == "comparator"].iterrows()]
        for name, s in lines:
            print(f"  {name:40s}" + "".join(pct(s[c]) for c in cols))

    print("\nImages on which at least 15 of the 17 valid configurations disagree with the experts")
    for ds in ["plantdoc", "plantwild"]:
        pool = man[(man.dataset == ds) & (man.set == "training_pool") & man.fewshot_exemplar_k.isna()].set_index("image_id")
        m = meta[(meta.dataset == ds) & meta.in_summaries & meta.prompt.isin(VALID_PROMPTS) & meta.image_id.isin(pool.index)]
        W = m.pivot_table(index="image_id", columns=["model", "prompt"], values="vlm_decision", aggfunc="first")
        assert W.shape[1] == 17
        n_rem = (W == "remove").sum(1).reindex(pool.index)
        rule = pool.expert_rule
        missed = rule[(rule != "none") & (n_rem <= 2)]
        over = rule[(rule == "none") & (n_rem >= 15)]
        print(f"  {ds}: removed by the experts and kept by >= 15: {len(missed)} of {(rule != 'none').sum()} "
              f"({missed.value_counts().to_dict()}), kept by the experts and removed by >= 15: {len(over)} of {(rule == 'none').sum()}")


if __name__ == "__main__":
    main()
