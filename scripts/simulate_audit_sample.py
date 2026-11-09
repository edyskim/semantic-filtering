"""How many expert-labeled images does the configuration selection rule need?

For each dataset, the script draws random audit samples of 50 to 500 training-pool images (without the few-shot
exemplars), selects the VLM configuration with the highest Cohen's kappa against the expert decisions on the sample,
and compares the choice with the selection on the full pool. It also records the noise share estimated from each
sample and the downstream accuracy difference of the selected configuration (natural setting, curated test set,
mean over the three classifiers). Results are written to summaries/audit_sample.csv.

usage: python scripts/simulate_audit_sample.py            (run from the repository root)
"""
import numpy as np
import pandas as pd

SIZES, DRAWS, SEED = [50, 100, 200, 300, 500], 2000, 20261104
VALID_PROMPTS = {"zero_shot", "few_shots", "cot", "few_shots_fixed"}


def kappa(e, V):
    """Cohen's kappa of every column of V (0/1 remove decisions) against the expert decisions e."""
    po = (V == e[:, None]).mean(0)
    pe = e.mean() * V.mean(0) + (1 - e.mean()) * (1 - V.mean(0))
    return (po - pe) / (1 - pe)


def main():
    rng = np.random.default_rng(SEED)
    meta = pd.read_csv("metadata/vlm_filtering_metadata.csv")
    man = pd.read_csv("manifests/split_manifest.csv")
    cond = pd.read_csv("summaries/downstream_conditions.csv")
    rows = []
    for ds in ["plantdoc", "plantwild"]:
        ex = set(man[(man.dataset == ds) & (man.set == "training_pool") & man.fewshot_exemplar_k.notna()].image_id)
        m = meta[(meta.dataset == ds) & meta.prompt.isin(VALID_PROMPTS) & meta.in_summaries & ~meta.image_id.isin(ex)]
        W = m.pivot_table(index="image_id", columns=["model", "prompt"], values="vlm_decision", aggfunc="first")
        e = (m.groupby("image_id").expert_decision.first().loc[W.index] == "remove").astype(int).values
        V = (W == "remove").astype(int).values
        cfgs = list(W.columns)
        full = kappa(e, V)
        best = int(np.argmax(full))
        g = cond[(cond.dataset == ds) & (cond.target_N.isna())]
        unf = g[(g.type == "baseline") & (g.model == "unfiltered")].set_index("arch").acc_mean
        gain = np.array([np.mean([g[(g.model == mo) & (g.prompt == pr) & (g.arch == a)].acc_mean.iloc[0] - unf[a] for a in unf.index])
                         for mo, pr in cfgs])
        print(f"{ds}: {len(e)} pool images, {len(cfgs)} configurations, full-pool choice {cfgs[best]} (kappa {full[best]:.3f})")
        for n in SIZES:
            sel, share = np.empty(DRAWS, int), np.empty(DRAWS)
            for r in range(DRAWS):
                idx = rng.choice(len(e), n, replace=False)
                sel[r] = int(np.nanargmax(kappa(e[idx], V[idx])))
                share[r] = e[idx].mean()
            loss = full[best] - full[sel]
            rows.append(dict(dataset=ds, audit_n=n, draws=DRAWS, same_as_full_pool=np.mean(sel == best),
                             kappa_loss_median=np.median(loss), kappa_loss_p90=np.percentile(loss, 90),
                             within_005_of_best=np.mean(loss <= 0.05),
                             noise_share_full=e.mean(), noise_share_lo=np.percentile(share, 2.5), noise_share_hi=np.percentile(share, 97.5),
                             gain_selected_median=np.median(gain[sel]), gain_selected_p10=np.percentile(gain[sel], 10),
                             gain_full_pool_choice=gain[best], gain_median_configuration=np.median(gain)))
    out = pd.DataFrame(rows)
    out.to_csv("summaries/audit_sample.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
