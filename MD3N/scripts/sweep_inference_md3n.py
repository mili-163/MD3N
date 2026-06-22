import argparse
from itertools import product
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


def metric(preds, labels):
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    non_zero = labels != 0
    acc = accuracy_score(labels[non_zero] > 0, preds[non_zero] > 0)
    f1 = f1_score(labels[non_zero] > 0, preds[non_zero] > 0, average="weighted")
    mae = float(np.mean(np.abs(preds - labels)))
    return acc, f1, mae


def make_args(mr, batch_size, overrides):
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
    args.update(overrides)
    # For one-point sweeps, force the scheduled value to the requested value.
    if "init_noise_scale" in overrides:
        args.pop("init_noise_scale_by_mr", None)
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
    return metric(preds, labels)


def parse_float_list(text):
    return [float(item) for item in text.split(",") if item]


def parse_int_list(text):
    return [int(item) for item in text.split(",") if item]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrs", default="0.1,0.7")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--init-noise", default="0.0,0.5,1.0")
    parser.add_argument("--guidance", default="0.0,0.05,0.1,0.2")
    parser.add_argument("--lambda-int", default="0.5,1.0,1.5")
    parser.add_argument("--sample-steps", default="8,12")
    args_cli = parser.parse_args()

    rows = []
    for mr, init_noise, guidance, lambda_int, sample_steps in product(
        parse_float_list(args_cli.mrs),
        parse_float_list(args_cli.init_noise),
        parse_float_list(args_cli.guidance),
        parse_float_list(args_cli.lambda_int),
        parse_int_list(args_cli.sample_steps),
    ):
        overrides = {
            "init_noise_scale": init_noise,
            "guidance_scale": guidance,
            "lambda_int_max": lambda_int,
            "sample_steps": sample_steps,
        }
        args = make_args(mr, args_cli.batch_size, overrides)
        model = load_model(args)
        acc, f1, mae = evaluate(args, model)
        row = (mr, init_noise, guidance, lambda_int, sample_steps, acc, f1, mae)
        rows.append(row)
        print(
            "MR=%.1f init=%.2f guidance=%.2f lint=%.2f steps=%d "
            "Acc2=%.4f F1=%.4f MAE=%.4f"
            % row
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("BEST_BY_MR")
    for mr in sorted(set(row[0] for row in rows)):
        best = max([row for row in rows if row[0] == mr], key=lambda item: item[5])
        print(
            "MR=%.1f best init=%.2f guidance=%.2f lint=%.2f steps=%d "
            "Acc2=%.4f F1=%.4f MAE=%.4f"
            % best
        )


if __name__ == "__main__":
    main()
