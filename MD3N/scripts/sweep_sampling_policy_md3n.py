import argparse
import random
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


def metrics(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    non_zero = labels != 0
    acc = accuracy_score(labels[non_zero] > 0, preds[non_zero] > 0)
    f1 = f1_score(labels[non_zero] > 0, preds[non_zero] > 0, average="weighted")
    mae = float(np.mean(np.abs(preds - labels)))
    return acc, f1, mae


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


def load_model(args, text_weight):
    setup_seed(1111)
    model = MD3N(args).to(args.device)
    origin_model = torch.load(args.pretrained_state_path, map_location=args.device)
    state = model.state_dict()
    state.update({key.replace("Model.", ""): value for key, value in origin_model.items()})
    model.load_state_dict(state, strict=False)
    model.eval()

    weights = [float(text_weight), 1.0, 1.0]

    def weighted_sampler(num_modal):
        pool = [0, 1, 2]
        chosen = []
        for _ in range(num_modal):
            total = sum(weights[idx] for idx in pool)
            draw = random.random() * total
            upto = 0.0
            for idx in pool:
                upto += weights[idx]
                if upto >= draw:
                    chosen.append(idx)
                    pool.remove(idx)
                    break
        return chosen, [idx for idx in [0, 1, 2] if idx not in chosen]

    model._sample_modalities = weighted_sampler
    return model


def evaluate(args, model):
    dataset = MMDataset(args, mode="test")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    miss_one, miss_two = 0, 0
    denominator = _missing_schedule_denominator(dataloader)
    drop_one, drop_two = _missing_probabilities(args)
    preds, labels = [], []
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
            preds.extend(outputs["M"].detach().cpu().view(-1).tolist())
            labels.extend(batch["labels"]["M"].view(-1).tolist())
    return metrics(preds, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrs", default="0.1,0.7")
    parser.add_argument("--text-weights", default="1,2,4,8,16")
    parser.add_argument("--batch-size", type=int, default=32)
    args_cli = parser.parse_args()

    for mr in [float(item) for item in args_cli.mrs.split(",") if item]:
        args = make_args(mr, args_cli.batch_size)
        for text_weight in [float(item) for item in args_cli.text_weights.split(",") if item]:
            model = load_model(args, text_weight)
            acc, f1, mae = evaluate(args, model)
            print(
                f"MR={mr:.1f} text_keep_weight={text_weight:.1f} "
                f"Acc2={acc:.4f} F1={f1:.4f} MAE={mae:.4f}"
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
