import argparse
from collections import defaultdict
from pathlib import Path
import sys
import types

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
COMBOS = {
    "T": [0],
    "V": [1],
    "A": [2],
    "T+V": [0, 1],
    "T+A": [0, 2],
    "V+A": [1, 2],
    "T+V+A": [0, 1, 2],
}


def binary_metrics(preds, labels):
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    non_zero = labels != 0
    if non_zero.any():
        acc = accuracy_score(labels[non_zero] > 0, preds[non_zero] > 0)
        f1 = f1_score(labels[non_zero] > 0, preds[non_zero] > 0, average="weighted")
    else:
        acc, f1 = 0.0, 0.0
    mae = float(np.mean(np.abs(preds - labels)))
    return acc, f1, mae, int(non_zero.sum())


def is_wrong(row):
    return row["label"] != 0 and ((row["pred"] > 0) != (row["label"] > 0))


def summarize(name, rows, top_k=8):
    preds = [row["pred"] for row in rows]
    labels = [row["label"] for row in rows]
    acc, f1, mae, non_zero = binary_metrics(preds, labels)
    wrong = [row for row in rows if is_wrong(row)]
    print(
        f"SUMMARY {name}: n={len(rows)} nz={non_zero} wrong={len(wrong)} "
        f"Acc2={acc:.4f} F1={f1:.4f} MAE={mae:.4f}"
    )

    buckets = [
        ("near_zero", lambda row: abs(row["label"]) <= 0.5),
        ("medium", lambda row: 0.5 < abs(row["label"]) <= 1.5),
        ("strong", lambda row: abs(row["label"]) > 1.5),
        ("true_pos", lambda row: row["label"] > 0),
        ("true_neg", lambda row: row["label"] < 0),
    ]
    for title, predicate in buckets:
        subset = [row for row in rows if predicate(row)]
        if not subset:
            continue
        b_acc, _, b_mae, b_non_zero = binary_metrics(
            [row["pred"] for row in subset],
            [row["label"] for row in subset],
        )
        print(
            f"  bucket {title}: n={len(subset)} nz={b_non_zero} "
            f"wrong={sum(is_wrong(row) for row in subset)} Acc2={b_acc:.4f} MAE={b_mae:.4f}"
        )

    by_pattern = defaultdict(list)
    for row in rows:
        by_pattern[row["pattern"]].append(row)
    for pattern in sorted(by_pattern):
        subset = by_pattern[pattern]
        p_acc, _, p_mae, p_non_zero = binary_metrics(
            [row["pred"] for row in subset],
            [row["label"] for row in subset],
        )
        print(
            f"  pattern {pattern}: n={len(subset)} nz={p_non_zero} "
            f"wrong={sum(is_wrong(row) for row in subset)} Acc2={p_acc:.4f} MAE={p_mae:.4f}"
        )

    print("  top_wrong:")
    for row in sorted(wrong, key=lambda item: abs(item["pred"] - item["label"]), reverse=True)[:top_k]:
        text = str(row["raw_text"]).replace("\n", " ")[:90]
        print(
            f"    idx={row['index']} id={row['id']} pat={row['pattern']} "
            f"y={row['label']:.3f} pred={row['pred']:.3f} text={text}"
        )


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


def append_rows(rows, batch, outputs, pattern):
    preds = outputs["M"].detach().cpu().view(-1).numpy()
    labels = batch["labels"]["M"].view(-1).numpy()
    for i, (pred, label) in enumerate(zip(preds, labels)):
        rows.append(
            {
                "pred": float(pred),
                "label": float(label),
                "pattern": pattern,
                "raw_text": batch["raw_text"][i],
                "id": batch["id"][i],
                "index": int(batch["index"][i]),
            }
        )


def eval_current_mr(args, dataloader, model):
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
            append_rows(rows, batch, outputs, pattern)
    summarize(f"current_mr={args.mr}", rows)


def eval_forced_combos(args, dataloader, model):
    for combo_name, available in COMBOS.items():
        def forced_sampler(self, num_modal, available=available):
            return list(available), [idx for idx in [0, 1, 2] if idx not in available]

        model._sample_modalities = types.MethodType(forced_sampler, model)
        rows = []
        setup_seed(1111)
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(
                    batch["text"].to(args.device),
                    batch["audio"].to(args.device),
                    batch["vision"].to(args.device),
                    num_modal=len(available),
                )
                append_rows(rows, batch, outputs, combo_name)
        summarize(f"forced_{combo_name}", rows, top_k=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--forced-combos", action="store_true")
    args_cli = parser.parse_args()

    args = make_args(args_cli.mr, args_cli.batch_size)
    dataset = MMDataset(args, mode="test")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = load_model(args)
    print(f"device={args.device} samples={len(dataset)} mr={args.mr}")
    eval_current_mr(args, dataloader, model)
    if args_cli.forced_combos:
        eval_forced_combos(args, dataloader, model)


if __name__ == "__main__":
    main()
