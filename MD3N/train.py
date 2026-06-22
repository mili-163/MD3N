"""
Training entrypoint for MD3N.

Examples:
  python train.py
  python train.py --smoke-test
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from config import get_config_regression
from run import MD3N_run
from trains.singleTask.model import scoremodel
from trains.singleTask.model.md3n import MD3N


DEFAULT_SEEDS = [1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119]
DATASET_DIMS = {
    "mosi": (768, 74, 35),
    "mosei": (768, 74, 35),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train MD3N or run a local smoke test.")
    parser.add_argument("--dataset", choices=sorted(DATASET_DIMS.keys()), default="mosi")
    parser.add_argument("--mr", type=float, default=0.1, help="Missing rate from 0.1 to 0.7.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny CPU-friendly sanity check with generated mock data.",
    )
    return parser.parse_args()


def _build_mock_split(num_samples, text_dim, audio_dim, vision_dim, seed, split_name):
    rng = np.random.default_rng(seed)
    seq_len = 50
    return {
        "text": rng.normal(size=(num_samples, seq_len, text_dim)).astype(np.float32),
        "audio": rng.normal(size=(num_samples, seq_len, audio_dim)).astype(np.float32),
        "vision": rng.normal(size=(num_samples, seq_len, vision_dim)).astype(np.float32),
        "raw_text": [f"{split_name}-{idx}" for idx in range(num_samples)],
        "id": [f"{split_name}-{idx}" for idx in range(num_samples)],
        "regression_labels": rng.uniform(-3.0, 3.0, size=(num_samples,)).astype(np.float32),
    }


def ensure_smoke_dataset(dataset_name, dataset_path):
    if dataset_path.is_file():
        return
    text_dim, audio_dim, vision_dim = DATASET_DIMS[dataset_name]
    dataset = {
        "train": _build_mock_split(2, text_dim, audio_dim, vision_dim, seed=0, split_name="train"),
        "valid": _build_mock_split(1, text_dim, audio_dim, vision_dim, seed=1, split_name="valid"),
        "test": _build_mock_split(1, text_dim, audio_dim, vision_dim, seed=2, split_name="test"),
    }
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "wb") as file_obj:
        pickle.dump(dataset, file_obj)


def ensure_smoke_weights(dataset_name, pretrained_path):
    if pretrained_path.is_file():
        return
    args = get_config_regression("md3n", dataset_name)
    args.update(
        {
            "use_bert": False,
            "use_finetune": False,
            "train_mode": "regression",
            "device": torch.device("cpu"),
        }
    )
    model = MD3N(args)
    state_dict = {f"Model.{key}": value.cpu() for key, value in model.state_dict().items()}
    pretrained_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, pretrained_path)


def run_smoke_test(dataset_name, mr):
    smoke_root = Path("dataset") / "smoke"
    feature_path = smoke_root / f"{dataset_name}_aligned_50.pkl"
    model_root = Path("pt") / "smoke"
    pretrained_path = model_root / f"pretrained-{dataset_name}.pth"

    ensure_smoke_dataset(dataset_name, feature_path)
    ensure_smoke_weights(dataset_name, pretrained_path)
    scoremodel.num_steps = 5

    MD3N_run(
        model_name="md3n",
        dataset_name=dataset_name,
        seeds=[1111],
        mr=mr,
        config={
            "featurePath": str(feature_path),
            "use_bert": False,
            "use_finetune": False,
            "batch_size": 2,
            "early_stop": 0,
            "patience": 1,
            "weight_decay": 0.0,
            "sample_steps": 4,
        },
        model_save_dir=str(model_root),
        res_save_dir="./result/smoke",
        log_dir="./log/smoke",
        gpu_ids=[],
        num_workers=0,
        verbose_level=1,
    )


def main():
    args = parse_args()
    if args.smoke_test:
        run_smoke_test(args.dataset, args.mr)
        return

    MD3N_run(
        model_name="md3n",
        dataset_name=args.dataset,
        seeds=args.seeds,
        mr=args.mr,
    )


if __name__ == "__main__":
    main()
