import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

__all__ = ['BertTextEncoder']

TRANSFORMERS_MAP = {
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
}


def _local_pretrained_candidates(pretrained):
    env_path = os.environ.get("TRANSFORMERS_LOCAL_PATH") or os.environ.get("BERT_LOCAL_PATH")
    if env_path:
        yield env_path

    yield pretrained

    cache_name = f"models--{pretrained.replace('/', '--')}"
    snapshot_root = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
    if snapshot_root.exists():
        for snapshot_dir in sorted(snapshot_root.iterdir(), reverse=True):
            if snapshot_dir.is_dir():
                yield str(snapshot_dir)


class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=False, transformers='bert', pretrained='bert-base-uncased'):
        super().__init__()

        tokenizer_class = TRANSFORMERS_MAP[transformers][1]
        model_class = TRANSFORMERS_MAP[transformers][0]
        load_error = None
        for local_pretrained in _local_pretrained_candidates(pretrained):
            try:
                # Prefer local snapshots so server training does not depend on HF connectivity.
                self.tokenizer = tokenizer_class.from_pretrained(local_pretrained, local_files_only=True)
                self.model = model_class.from_pretrained(local_pretrained, local_files_only=True)
                break
            except OSError as exc:
                load_error = exc
        else:
            if os.environ.get("TRANSFORMERS_OFFLINE") or os.environ.get("HF_HUB_OFFLINE"):
                raise load_error
            self.tokenizer = tokenizer_class.from_pretrained(pretrained)
            self.model = model_class.from_pretrained(pretrained)
        self.use_finetune = use_finetune
    
    def get_tokenizer(self):
        return self.tokenizer
    
    # def from_text(self, text):
    #     """
    #     text: raw data
    #     """
    #     input_ids = self.get_id(text)
    #     with torch.no_grad():
    #         last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
    #     return last_hidden_states.squeeze()
    
    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states
