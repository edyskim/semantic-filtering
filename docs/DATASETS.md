# Dataset sources

## PlantDoc

PlantDoc is from:

Davinder Singh, Naman Jain, Pranjali Jain, Pratik Kayal, Sudhakar Kumawat, and Nipun Batra. PlantDoc: A Dataset for Visual Plant Disease Detection. CoDS-COMAD 2020.

- Paper DOI: https://doi.org/10.1145/3371158.3371196
- Dataset repository: https://github.com/pratikkayal/PlantDoc-Dataset

In this study, PlantDoc is used as a real-world plant disease field dataset. The experiments use a controlled top-10-class subset of 1,000 images for stable downstream comparison.

## PlantWild

PlantWild is from:

Tianqi Wei, Zhi Chen, Zi Huang, and Xin Yu. Benchmarking In-the-Wild Multimodal Disease Recognition and A Versatile Baseline. ACM Multimedia 2024.

- Project page: https://tqwei05.github.io/PlantWild/
- Dataset: https://huggingface.co/datasets/uqtwei2/PlantWild
- Baseline code: https://github.com/tqwei05/MVPDR

In this study, PlantWild is used as an in-the-wild plant disease dataset. The dataset card reports 18,542 images across 89 classes. The experiments use a controlled top-10-class subset of 2,000 images for stable downstream comparison.

## Data note

This repository does not redistribute original images. The released metadata is an
annotation layer (image identifiers, class labels, keep/remove decisions, prompt
condition, and rule attribution) and contains no image files.

Users should obtain raw images directly from the original PlantDoc and PlantWild
sources above.
