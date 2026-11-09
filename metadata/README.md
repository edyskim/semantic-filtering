# Public filtering metadata

This directory contains decision-level metadata from the semantic filtering experiments. It supports verification of every filtering decision without redistributing raw images or full model responses.

## Files

| File | Rows | Main columns | Purpose |
|---|---:|---|---|
| `expert_reference_metadata.csv` | 3,000 | `dataset`, `image_id`, `class_name`, `label`, `expert_decision`, `expert_rule` | One row per training-pool image with the expert reference decision and rule |
| `vlm_filtering_metadata.csv` | 93,000 | `dataset`, `image_id`, `class_name`, `label`, `model`, `prompt`, `expert_decision`, `expert_rule`, `vlm_decision`, `vlm_rule`, `in_summaries` | One row per image, model, and prompt condition |
| `comparator_decisions.csv` | 15,000 | `dataset`, `image_id`, `method`, `decision` | Keep/remove decision of each automatic comparator (CleanLab, DeFT, CLIP, VisCE2 with GPT-4o-mini and with Gemini-2.0-Flash-Lite) for every training-pool image |

`image_id` links both files to `../manifests/split_manifest.csv`, which gives the source path and SHA-256 checksum of each image.

The decision columns use `keep` and `remove`. The rule columns use `none` for kept images and `rule1` to `rule5` for removed images. The rule-free prompt contains no rules, so its removals carry `unspecified`.

## Prompt conditions

| `prompt` | Models | Description |
|---|---|---|
| `zero_shot`, `few_shots`, `cot` | all six | Main prompt formats |
| `few_shots_fixed` | Gemini-2.5-Flash | Few-shot re-run with correctly labeled exemplars |
| `rule_free` | GPT-4o-mini, GPT-4o, Claude-Sonnet-4.6, Claude-Haiku-4.5 | Same task and output format without the expert rules |
| `sc5_cot` | Claude-Sonnet-4.6, Claude-Haiku-4.5 | Self-consistency: five CoT samples at temperature 0.7, majority decision |
| `zero_shot_repeat` | GPT-4o-mini, GPT-4o, Claude-Haiku-4.5 | Repeated run of the unchanged zero-shot prompt |
| `zero_shot_reversed` | Claude-Haiku-4.5 | Rule blocks listed in reverse order |
| `few_shots_k2`, `few_shots_k3` | Claude-Haiku-4.5 | Two and three exemplars per category |

The verbatim prompt texts are in `../prompts/verbatim/`, and model identifiers and access dates are in `../docs/MODELS.md`.

`in_summaries` is `False` for the original few-shot runs of the two Gemini models, which displayed the accepted exemplar with the decision REMOVE. These rows are kept for completeness and excluded from all summary statistics.

## Excluded fields

The metadata intentionally excludes raw images, local image paths, full natural-language model responses, API logs, model checkpoints, and credentials. Raw images should be obtained from the original PlantDoc and PlantWild sources listed in `../docs/DATASETS.md`.

## License

This metadata is licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). See `../LICENSE-DATA.md` for the terms and suggested attribution.
