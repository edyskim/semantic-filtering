"""Recompute the downstream statistics of the article from the released per-image test predictions.

For every condition (dataset, filter, prompt, classifier, setting) the script averages the correctness of each curated
test image over the five seeds, computes a bootstrap 95% interval of accuracy (10,000 resamples), compares the
condition with unfiltered training of the same dataset, classifier, and setting through an exact two-sided paired
sign-flip test, and adjusts the p-values with the Holm procedure within each dataset, classifier, and setting.
It then prints the summary rows of the main-text downstream tables and checks every number against
summaries/downstream_conditions.csv.

usage: python scripts/reproduce_downstream_tables.py            (run from the repository root)
"""
import zlib
import numpy as np
import pandas as pd

B, N_SEEDS = 10000, 5
EXCLUDED = {("gemini-2.0-flash-lite", "few_shots"), ("gemini-2.5-flash", "few_shots")}
VALID_PROMPTS = {"zero_shot", "few_shots", "cot", "few_shots_fixed"}
cond_rng = lambda name, tag: np.random.default_rng([20261103, zlib.crc32(f"{name}|{tag}".encode())])


def boot_ci(x, rng):
    m = x[rng.integers(0, len(x), size=(B, len(x)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def signflip_p(d):
    """Exact paired sign-flip test. Seed-averaged differences are multiples of 1/N_SEEDS, so the null
    distribution of their sum over all 2^n sign assignments follows by convolution."""
    k = np.rint(d * N_SEEDS)
    assert np.allclose(k, d * N_SEEDS)
    k = np.abs(k[k != 0]).astype(int)
    if len(k) == 0:
        return 1.0
    K = int(k.sum())
    dist = np.zeros(2 * K + 1); dist[K] = 1.0
    for x in k:
        dist = 0.5 * (np.roll(dist, x) + np.roll(dist, -x))
    return float(min(1.0, dist[np.abs(np.arange(-K, K + 1)) >= abs(np.rint(d * N_SEEDS).sum())].sum()))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def main():
    sd = pd.read_csv("downstream/seed_results.csv", keep_default_na=False)
    pred = pd.concat([pd.read_csv(f"predictions/{ds}_test_predictions.csv.gz") for ds in ["plantdoc", "plantwild"]])
    pred = pred[pred.test_set == "curated"]
    pred["correct"] = (pred.true_class == pred.pred_class).astype(float)
    corr = {c: g.pivot_table(index="image_id", columns="seed", values="correct", sort=False).mean(1) for c, g in pred.groupby("condition", sort=False)}
    dup = pd.read_csv("audits/exact_duplicates.csv")
    dup = set(zip(dup[dup.test_set == "curated test"].dataset, dup[dup.test_set == "curated test"].test_image_id))

    cond = sd.groupby("condition").agg(dataset=("dataset", "first"), filter_type=("filter_type", "first"), filter=("filter", "first"),
                                       prompt=("prompt", "first"), classifier=("classifier", "first"), setting=("setting", "first"),
                                       in_summaries=("in_summaries", "first"), acc_mean=("accuracy", "mean")).reset_index()
    recs = []
    for (ds, arch, st), g in cond.groupby(["dataset", "classifier", "setting"]):
        base = g[(g.filter_type == "baseline") & (g["filter"] == "unfiltered")].condition.iloc[0]
        cb = corr[base]
        for _, r in g.iterrows():
            c = corr[r.condition]
            assert (c.index == cb.index).all()
            rec = r.to_dict()
            rec["acc_ci_lo"], rec["acc_ci_hi"] = boot_ci(c.values, cond_rng(r.condition, "acc"))
            if r.condition != base:
                d = (c - cb).values
                keep = ~np.array([(ds, i) in dup for i in c.index])
                rec.update(diff_mean=d.mean(), p_perm=signflip_p(d), nodup_diff=d[keep].mean())
            recs.append(rec)
    out = pd.DataFrame(recs)
    out["p_holm"] = np.nan
    for _, idx in out[out.p_perm.notna()].groupby(["dataset", "classifier", "setting"]).groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_perm"].values)

    # check against the released summary
    ref = pd.read_csv("summaries/downstream_conditions.csv").set_index("name")
    chk = out.set_index("condition").join(ref, rsuffix="_ref")
    for a, b in [("acc_mean", "acc_mean_ref"), ("acc_ci_lo", "acc_ci_lo_ref"), ("acc_ci_hi", "acc_ci_hi_ref"),
                 ("diff_mean", "diff_mean_ref"), ("p_holm", "p_holm_ref"), ("nodup_diff", "nodup_diff_ref")]:
        dev = (chk[a] - chk[b]).abs().max()
        print(f"{a:10s} max deviation from released summary: {dev:.2e}")

    # summary rows of the main-text tables (VLM configurations used in the summaries)
    print("\nVLM configurations against unfiltered training (curated test sets)")
    print(f"{'dataset':10s} {'classifier':10s} {'setting':10s} {'unfilt.':>8s} {'median':>8s} {'above':>6s} {'sig+':>5s} {'sig-':>5s}")
    v = out[(out.filter_type == "vlm") & out.in_summaries.astype(str).isin(["True", "true", "1"])]
    for (ds, arch, st), g in v.groupby(["dataset", "classifier", "setting"]):
        unf = out[(out.dataset == ds) & (out.classifier == arch) & (out.setting == st) & (out["filter"] == "unfiltered")].acc_mean.iloc[0]
        print(f"{ds:10s} {arch:10s} {st:10s} {unf:8.3f} {g.acc_mean.median():8.3f} {int((g.acc_mean > unf).sum()):3d}/{len(g):<2d} "
              f"{int(((g.p_holm < 0.05) & (g.diff_mean > 0)).sum()):5d} {int(((g.p_holm < 0.05) & (g.diff_mean < 0)).sum()):5d}")


if __name__ == "__main__":
    main()
