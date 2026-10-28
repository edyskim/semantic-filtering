# Data directory

Raw images are not included in this repository.

Download PlantDoc and PlantWild from their original sources listed in `docs/DATASETS.md`, then prepare local CSV manifests for the scripts.

Recommended local layout:

```text
data/raw/plantdoc/
data/raw/plantwild/
data/processed/plantdoc/
data/processed/plantwild/
```

For VLM filtering, use a manifest with:

```text
img_path,class_name
```

For downstream classifier training, use a manifest with:

```text
img_path,class_name,label
```

`img_path` should point to the local image file on your machine. Do not commit raw images, local absolute paths, or generated API outputs.
