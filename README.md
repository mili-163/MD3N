# MD3N

This repository provides the PyTorch implementation used for the paper submission. The code supports incomplete multimodal sentiment experiments on aligned MOSI/MOSEI-style features.

## Environment

The code was tested with Python, PyTorch, CUDA, NumPy, scikit-learn, pandas, tqdm, and transformers. A CUDA-enabled GPU is recommended for the full experiments.

## Data and Checkpoints

Datasets are not included in this code release. 

The code expects each split to provide language, acoustic, visual, label, and sample-id fields in the format used by the dataloader. The default feature dimensions follow the paper setting: language `768`, visual `35`, and acoustic `74`; the dataloader can also infer dimensions from the provided feature file for compatible preprocessed variants.

Pretrained checkpoints, if used, should be specified through the configuration or command-line entry point.


