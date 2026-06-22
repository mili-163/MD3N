import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.MD3N import _missing_probabilities, _missing_schedule_denominator
from trains.singleTask.model.md3n import MD3N
from utils import setup_seed


MODALITIES = {0: "T", 1: "V", 2: "A"}


def make_args(mr, batch_size):
    args = get_config_regression("md3n", "mosi")
    args.mode = "test"
    args.mr = mr
    args.update(
        {
            "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            "train_mode": "regression",
            "feature_T": "",
            "feature_A": "",
            "feature_V": "",
            "model_name": "md3n",
            "dataset_name": "mosi",
            "pretrained_state_path": Path("pt/pretrained-mosi.pth"),
            "batch_size": batch_size,
        }
    )
    return args


def load_model(args):
    setup_seed(1111)
    model = MD3N(args).to(args.device)
    origin_model = torch.load(args.pretrained_state_path, map_location=args.device)
    state = model.state_dict()
    state.update({key.replace("Model.", ""): value for key, value in origin_model.items()})
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def collect_rows(args, split, model):
    dataset = MMDataset(args, mode=split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = []
    miss_one, miss_two = 0, 0
    denominator = _missing_schedule_denominator(dataloader)
    drop_one, drop_two = _missing_probabilities(args)
    setup_seed(1111)
    with torch.no_grad():
        for batch in dataloader:
            if miss_two / denominator < drop_two[int(args.mr * 10 - 1)]:
                num_modal = 1
                miss_two += 1
            elif miss_one / denominator < drop_one[int(args.mr * 10 - 1)]:
                num_modal = 2
                miss_one += 1
            else:
                num_modal = 3
            outputs = model(
                batch["text"].to(args.device),
                batch["audio"].to(args.device),
                batch["vision"].to(args.device),
                num_modal=num_modal,
            )
            pattern = "+".join(MODALITIES[idx] for idx in sorted(outputs["ava_modal_idx"]))
            preds = outputs["M"].detach().cpu().view(-1).numpy()
            labels = batch["labels"]["M"].view(-1).numpy()
            for pred, label in zip(preds, labels):
                rows.append({"pred": float(pred), "label": float(label), "pattern": pattern})
    return rows


def acc_at_threshold(preds, labels, threshold):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    non_zero = labels != 0
    if not non_zero.any():
        return 0.0
    return accuracy_score(labels[non_zero] > 0, preds[non_zero] > threshold)


def metrics(rows, thresholds=None):
    thresholds = thresholds or {}
    preds = np.asarray([row["pred"] - thresholds.get(row["pattern"], thresholds.get("__global__", 0.0)) for row in rows])
    labels = np.asarray([row["label"] for row in rows])
    non_zero = labels != 0
    acc = accuracy_score(labels[non_zero] > 0, preds[non_zero] > 0)
    f1 = f1_score(labels[non_zero] > 0, preds[non_zero] > 0, average="weighted")
    mae = float(np.mean(np.abs(preds - labels)))
    return acc, f1, mae


def best_threshold(rows):
    labels = np.asarray([row["label"] for row in rows])
    preds = np.asarray([row["pred"] for row in rows])
    non_zero = labels != 0
    if non_zero.sum() < 4:
        return 0.0, acc_at_threshold(preds, labels, 0.0)
    preds_nz = preds[non_zero]
    candidates = np.unique(np.concatenate(([preds_nz.min() - 1e-4, 0.0, preds_nz.max() + 1e-4], preds_nz)))
    best_t, best_acc = 0.0, acc_at_threshold(preds, labels, 0.0)
    for threshold in candidates:
        cur = acc_at_threshold(preds, labels, threshold)
        if cur > best_acc:
            best_t, best_acc = float(threshold), float(cur)
    return best_t, best_acc


def fit_thresholds(valid_rows, min_count):
    thresholds = {}
    global_threshold, global_acc = best_threshold(valid_rows)
    thresholds["__global__"] = global_threshold
    by_pattern = {}
    for row in valid_rows:
        by_pattern.setdefault(row["pattern"], []).append(row)
    print(f"global threshold={global_threshold:.4f} valid_acc={global_acc:.4f}")
    for pattern, rows in sorted(by_pattern.items()):
        threshold, acc = best_threshold(rows)
        use_threshold = threshold if len(rows) >= min_count else global_threshold
        thresholds[pattern] = use_threshold
        print(
            f"pattern={pattern} n={len(rows)} threshold={threshold:.4f} "
            f"used={use_threshold:.4f} valid_acc={acc:.4f}"
        )
    return thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-count", type=int, default=24)
    args_cli = parser.parse_args()

    args = make_args(args_cli.mr, args_cli.batch_size)
    model = load_model(args)
    valid_rows = collect_rows(args, "valid", model)
    test_rows = collect_rows(args, "test", model)
    raw_valid = metrics(valid_rows)
    raw_test = metrics(test_rows)
    thresholds = fit_thresholds(valid_rows, args_cli.min_count)
    cal_valid = metrics(valid_rows, thresholds)
    cal_test = metrics(test_rows, thresholds)
    print(
        f"RAW valid Acc2={raw_valid[0]:.4f} F1={raw_valid[1]:.4f} MAE={raw_valid[2]:.4f} | "
        f"test Acc2={raw_test[0]:.4f} F1={raw_test[1]:.4f} MAE={raw_test[2]:.4f}"
    )
    print(
        f"CAL valid Acc2={cal_valid[0]:.4f} F1={cal_valid[1]:.4f} MAE={cal_valid[2]:.4f} | "
        f"test Acc2={cal_test[0]:.4f} F1={cal_test[1]:.4f} MAE={cal_test[2]:.4f}"
    )


if __name__ == "__main__":
    main()
