# Summary results

| File | Content |
|---|---|
| `filter_metrics.csv` | Precision, recall, F1, Cohen's kappa, agreement, and rule-attribution accuracy of every configuration against the expert reference labels, on all pool images and without the few-shot exemplar images (`exexemplar_*` columns, used in the article) |
| `rationale_consistency.csv` | Agreement of the cited rule between prompt formats of the same model and with the expert rule |
| `noise_composition.csv` | Expert keep/remove composition of each training pool by rule |
| `cost_measured.csv` | Measured tokens per image and cost per 1,000 images |
| `latency_may2026.csv` | Measured latency per image |
| `downstream_conditions.csv` | One row per downstream condition: seed means and SDs of accuracy, macro-F1, AUROC, AUPRC, and uncurated-test accuracy, bootstrap 95% interval of accuracy, paired difference to unfiltered training with its interval, exact sign-flip p-value and Holm-adjusted p-value, and the same difference without the duplicated test images |
| `vlm_distribution.csv` | Distribution of accuracy over the VLM configurations used in the summaries, per dataset, classifier, and setting |
| `kappa_selected.csv` | Downstream results of the configuration with the highest kappa on the training pool |
| `classwise.csv` | Seed-averaged recall of every class for every natural-setting condition |
| `rule_detection.csv` | Share of the images removed by the experts under each rule that each VLM configuration, rule-free and self-consistency run, and comparator also removed, and share of the images kept by the experts that it removed, without the few-shot exemplar images (`../scripts/reproduce_rule_detection.py`) |
| `audit_sample.csv` | Configuration selection and noise-share estimation from random expert-labeled audit samples of 50 to 500 images (`../scripts/simulate_audit_sample.py`) |

Filter-level values can be recomputed from `../metadata/` with `../scripts/reproduce_filter_tables.py`, and downstream values from `../predictions/` and `../downstream/` with `../scripts/reproduce_downstream_tables.py`.
