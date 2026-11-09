# Split manifest

`split_manifest.csv` lists every image used in the study, one row per image.

| Column | Meaning |
|---|---|
| `dataset` | `plantdoc` or `plantwild` |
| `set` | `training_pool` (1,000 PlantDoc and 2,000 PlantWild images) or `test` (official test split restricted to the 10 classes, 93 and 794 images) |
| `image_id` | Identifier used in all metadata, prediction, and audit files |
| `class_name`, `label` | Class name and integer label used for training and evaluation |
| `source_path` | Path of the image inside the original dataset release (`<split>/<class folder>/<file>`) |
| `sha256` | SHA-256 checksum of the image file, for verifying a local copy |
| `expert_decision`, `expert_rule` | Expert reference label of training-pool images |
| `in_curated_test` | `True` for the 85 PlantDoc and 345 PlantWild test images kept after expert curation (the curated test sets) |
| `fewshot_exemplar_k` | For exemplar images, the smallest number of exemplars per category at which the image is shown (1 is the main few-shot condition). These 18 images per dataset are excluded from all filtering metrics |

The training/validation division of each filtered pool is drawn inside the training script, holding out 10% of each class as validation data with a fixed seed (`random_state = 42`), so it is determined by the pool and needs no separate file.

When the source release contains several identical files, `source_path` gives one of them, preferring the same split and class. Exact duplicates between the test sets and the training pools are listed in `../audits/exact_duplicates.csv`.
