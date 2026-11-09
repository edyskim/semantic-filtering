# Experiment overview

This repository keeps the public code organized by the main analysis steps in the manuscript. It is intended to make the filtering workflow, prompt design, and metadata structure transparent without redistributing raw images or full API responses.

## Final protocol

```text
downstream epochs = 30
seeds = [0, 42, 84, 126, 168]
VLM temperature = 0.0
VLM query mode = stateless single-image requests
```

## Analysis steps

1. Define expert semantic filtering rules.
2. Convert those rules into zero-shot, few-shot, and chain-of-thought prompts.
3. Apply VLM filtering to image manifests.
4. Compare VLM keep/remove decisions with expert reference labels.
5. Analyze precision, recall, F1, agreement, and Cohen's kappa.
6. Train downstream CNN classifiers on retained subsets.
7. Run fixed-size diagnostic comparisons to control retained training-set size.
8. Compute inter-annotator agreement for expert reference labels.

## Public metadata

The public release includes decision-level metadata under `metadata/`. Manuscript
summary tables are not redistributed here.

```text
metadata/expert_reference_metadata.csv
metadata/vlm_filtering_metadata.csv
```

These files are decision-level metadata. They include image identifiers, dataset name, class name, integer class label, expert keep/remove decision, VLM keep/remove decision, prompt condition, model name, and rule attribution.

They intentionally exclude local image paths, raw model rationales, full API outputs, and raw images. `scripts/run_vlm_filtering.py` writes the same decision schema (`vlm_decision` in `keep`/`remove` and `vlm_rule` in `none`/`rule1`..`rule5`), so locally generated outputs use the same encoding as this metadata.

## Reproducing analyses

The scripts in `scripts/` are standalone entry points:

```text
compute_iaa.py
compute_filtering_metrics.py
make_fixed_size_subset.py
run_vlm_filtering.py
train_downstream_classifier.py
```

Users should prepare local CSV manifests from the original PlantDoc and PlantWild datasets before running VLM filtering or downstream classifier training.
