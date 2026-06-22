import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.model.md3n import MD3N
from utils import setup_seed


MODALITIES = {0: "L", 1: "V", 2: "A"}


def regression_metrics(preds, labels):
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    non_zero = labels != 0
    acc2 = accuracy_score(labels[non_zero] > 0, preds[non_zero] > 0)
    f1 = f1_score(labels[non_zero] > 0, preds[non_zero] > 0, average="weighted")
    acc7 = np.mean(np.round(np.clip(preds, -3, 3)) == np.round(np.clip(labels, -3, 3)))
    mae = float(np.mean(np.abs(preds - labels)))
    return {
        "Acc2": round(float(acc2) * 100, 2),
        "F1": round(float(f1) * 100, 2),
        "Acc7": round(float(acc7) * 100, 2),
        "MAE": round(mae, 4),
    }


def make_args(batch_size, latent_dim, nheads, sample_steps):
    args = get_config_regression("md3n", "mosi")
    args.mode = "test"
    args.mr = 0.7
    args.update(
        {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "train_mode": "regression",
            "feature_T": "",
            "feature_A": "",
            "feature_V": "",
            "model_name": "md3n",
            "dataset_name": "mosi",
            "batch_size": batch_size,
            "dst_feature_dim_nheads": [latent_dim, nheads],
            "sample_steps": sample_steps,
            "init_noise_scale": 0.0,
        }
    )
    args.pop("init_noise_scale_by_mr", None)
    return args


def prepare_dataset(args, split):
    dataset = MMDataset(args, mode=split)
    args.seq_lens = dataset.get_seq_len()
    return dataset


def load_model(args, checkpoint):
    setup_seed(1111)
    model = MD3N(args).to(args.device)
    raw_state = torch.load(checkpoint, map_location=args.device)
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in raw_state.items():
        name = key.replace("Model.", "")
        if name in model_state and tuple(model_state[name].shape) == tuple(value.shape):
            compatible[name] = value
        else:
            skipped.append(name)
    model_state.update(compatible)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    return model, {"loaded": len(compatible), "skipped": len(skipped)}


def force_available(model, available):
    def sampler(self, num_modal):
        return list(available), [idx for idx in [0, 1, 2] if idx not in available]

    model._sample_modalities = types.MethodType(sampler, model)


def evaluate_static_av(args, model, dataloader):
    force_available(model, [1, 2])
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                batch["text"].to(args.device),
                batch["audio"].to(args.device),
                batch["vision"].to(args.device),
                num_modal=2,
            )
            preds.extend(outputs["M"].detach().cpu().view(-1).tolist())
            labels.extend(batch["labels"]["M"].view(-1).tolist())
    return regression_metrics(preds, labels)


def evaluate_protocol3(args, model, dataloader, withheld_ratio):
    one_modal_patterns = [[0], [1], [2]]
    preds, labels = [], []
    pattern_counts = {}
    masked_targets = 0
    withheld_targets = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            available = one_modal_patterns[batch_idx % len(one_modal_patterns)]
            force_available(model, available)
            missing = [idx for idx in [0, 1, 2] if idx not in available]
            pattern = "+".join(MODALITIES[idx] for idx in available)
            batch_size = batch["labels"]["M"].size(0)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + batch_size
            masked_targets += batch_size * len(missing)
            withheld_targets += int(round(batch_size * len(missing) * withheld_ratio))
            outputs = model(
                batch["text"].to(args.device),
                batch["audio"].to(args.device),
                batch["vision"].to(args.device),
                num_modal=1,
            )
            preds.extend(outputs["M"].detach().cpu().view(-1).tolist())
            labels.extend(batch["labels"]["M"].view(-1).tolist())
    metrics = regression_metrics(preds, labels)
    metrics["PatternCounts"] = pattern_counts
    metrics["MaskedTargets"] = masked_targets
    metrics["WithheldTargets"] = withheld_targets
    metrics["WithheldRatio"] = round(withheld_targets / max(1, masked_targets), 4)
    metrics["EffectiveMissingRate"] = round(2 / 3, 4)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="pt/pretrained-mosi.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=12)
    parser.add_argument("--withheld-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--output", default="")
    cli = parser.parse_args()

    setup_seed(cli.seed)
    args = make_args(cli.batch_size, cli.latent_dim, cli.nheads, cli.sample_steps)
    dataset = prepare_dataset(args, "test")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model, load_stats = load_model(args, Path(cli.checkpoint))

    result = {
        "dataset": "MOSI",
        "checkpoint": cli.checkpoint,
        "samples": len(dataset),
        "device": str(args.device),
        "load_stats": load_stats,
        "protocol_ii_static_AV": evaluate_static_av(args, model, dataloader),
        "protocol_iii_natural": evaluate_protocol3(args, model, dataloader, cli.withheld_ratio),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if cli.output:
        output = Path(cli.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
