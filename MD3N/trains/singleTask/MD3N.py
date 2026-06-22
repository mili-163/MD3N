import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
logger = logging.getLogger('MMSA')


def _missing_schedule_denominator(dataloader):
    return max(1, int(np.round(len(dataloader) / 10) * 10))


def _missing_probabilities(args):
    drop_two = args.get('drop_two_probs', args.get('synthetic_drop_probs', [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]))
    drop_one = args.get('drop_one_probs', [0.1, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0])
    return drop_one, drop_two

class MD3N():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer_params = model.parameters()
        bert_learning_rate = self.args.get('bert_learning_rate', None)
        if bert_learning_rate is not None and hasattr(model, 'text_model'):
            bert_param_ids = {id(param) for param in model.text_model.parameters()}
            base_params = [
                param for param in model.parameters()
                if param.requires_grad and id(param) not in bert_param_ids
            ]
            bert_params = [param for param in model.text_model.parameters() if param.requires_grad]
            optimizer_params = [
                {'params': base_params, 'lr': self.args.learning_rate},
                {'params': bert_params, 'lr': float(bert_learning_rate)},
            ]
        optimizer = optim.AdamW(
            optimizer_params,
            lr=self.args.learning_rate,
            weight_decay=float(self.args.get('weight_decay', 0.0)),
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.args.patience)
        # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        # load pretrained
        pretrained_path = Path(self.args.pretrained_state_path)
        if not pretrained_path.is_file():
            raise FileNotFoundError(
                f"Missing pretrained weights: {pretrained_path}. "
                "Download the paper weights or run `python train.py --smoke-test`."
            )
        origin_model = torch.load(pretrained_path, map_location=self.args.device)
        net_dict = model.state_dict()
        new_state_dict = {}
        for k, v in origin_model.items():
            k = k.replace('Model.', '')
            new_state_dict[k] = v
        net_dict.update(new_state_dict)
        model.load_state_dict(net_dict, strict=False)

        while True:
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
            model.train()
            train_loss = 0.0
            miss_one, miss_two = 0, 0  # num of missing one modal and missing two modal
            left_epochs = self.args.update_epochs
            missing_denominator = _missing_schedule_denominator(dataloader['train'])
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    # forward
                    miss_1, miss_2 = _missing_probabilities(self.args)
                    if miss_two / missing_denominator < miss_2[int(self.args.mr*10-1)]:  # missing two modal
                        outputs = model(text, audio, vision, num_modal=1)
                        miss_two += 1
                    elif miss_one / missing_denominator < miss_1[int(self.args.mr*10-1)]:  # missing one modal
                        outputs = model(text, audio, vision, num_modal=2)
                        miss_one += 1
                    else:  # no missing
                        outputs = model(text, audio, vision, num_modal=3)

                    # compute loss
                    task_loss = self.criterion(outputs['M'], labels)
                    loss_score = outputs.get('loss_score', torch.zeros((), device=self.args.device))
                    loss_stage = outputs.get('loss_stage', torch.zeros((), device=self.args.device))
                    loss_end = outputs.get('loss_end', torch.zeros((), device=self.args.device))
                    non_zero = labels.view(-1) != 0
                    if non_zero.any():
                        loss_cls = F.binary_cross_entropy_with_logits(
                            outputs['M'].view(-1)[non_zero],
                            (labels.view(-1)[non_zero] > 0).float(),
                        )
                    else:
                        loss_cls = torch.zeros((), device=self.args.device)
                    stage_weight = float(outputs.get('stage_weight', 1.0))
                    combine_loss = (
                        task_loss
                        + float(self.args.get('gamma_score', 1.0)) * loss_score
                        + float(self.args.get('gamma_stage', 1.0)) * stage_weight * loss_stage
                        + float(self.args.get('gamma_end', 1.0)) * loss_end
                        + float(self.args.get('gamma_cls', 0.0)) * loss_cls
                    )

                    # backward
                    combine_loss.backward()
                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad],
                                                  self.args.grad_clip)
                    # store results
                    train_loss += combine_loss.item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            aux_summary = (
                f"score: {round(float(loss_score.detach().cpu()), 4)} "
                f"stage: {round(float(loss_stage.detach().cpu()), 4)} "
                f"end: {round(float(loss_end.detach().cpu()), 4)} "
                f"cls: {round(float(loss_cls.detach().cpu()), 4)} "
                f"stage_id: {outputs.get('stage_id', '-')}"
            )
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)} {aux_summary}"
            )
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            test_results = self.do_test(model, dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save each epoch model
            model_save_path = Path(self.args.checkpoint_dir) / f"{epochs}.pth"
            torch.save(model.state_dict(), model_save_path)
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # epoch results
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        miss_one, miss_two = 0, 0

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        with torch.no_grad():
            missing_denominator = _missing_schedule_denominator(dataloader)
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    miss_1, miss_2 = _missing_probabilities(self.args)
                    if miss_two / missing_denominator < miss_2[int(self.args.mr * 10 - 1)]:  # missing two modal
                        outputs = model(text, audio, vision, num_modal=1)
                        miss_two += 1
                    elif miss_one / missing_denominator < miss_1[int(self.args.mr * 10 - 1)]:  # missing one modal
                        outputs = model(text, audio, vision, num_modal=2)
                        miss_one += 1
                    else:  # no missing
                        outputs = model(text, audio, vision, num_modal=3)

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        for item in features.keys():
                            features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels.cpu().detach().tolist())
                        preds = outputs["M"].cpu().detach().numpy()
                        sample_results.extend(preds.squeeze())

                    loss = self.criterion(outputs['M'], labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results


IMDER = MD3N
