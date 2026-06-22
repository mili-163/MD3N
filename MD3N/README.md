# MD3N

This repository provides the PyTorch implementation used for the paper submission. The code supports incomplete multimodal sentiment experiments on aligned MOSI/MOSEI-style features.

## Environment

The code was tested with Python, PyTorch, CUDA, NumPy, scikit-learn, pandas, tqdm, and transformers. A CUDA-enabled GPU is recommended for the full experiments.

## Data and Checkpoints

Datasets are not included in this code release. Please prepare aligned multimodal features separately and set the corresponding paths in `config/config.json`.

The code expects each split to provide language, acoustic, visual, label, and sample-id fields in the format used by the dataloader. The default feature dimensions follow the paper setting: language `768`, visual `35`, and acoustic `74`; the dataloader can also infer dimensions from the provided feature file for compatible preprocessed variants.

Pretrained checkpoints, if used, should be specified through the configuration or command-line entry point.

## Running

Train/evaluate MD3N on MOSI:

```bash
python train.py --dataset mosi --mr 0.1
```

Run a quick sanity check without real data:

```bash
python train.py --smoke-test
```

Main hyperparameters are defined in `config/config.json`. Results and logs are written to `result/` and `log/`.

## Notes for Review

This code release is intended for reproducibility. Please update `config/config.json` when using different datasets, feature layouts, or checkpoint locations.
