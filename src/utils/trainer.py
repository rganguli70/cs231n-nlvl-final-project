from transformers import Trainer
from torch import nn
import torch

span_loss_fn = nn.MSELoss()

class NLVLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        label_ids = inputs.pop("label_ids")
        preds = model(**inputs)
        span_preds = preds[0].squeeze(1)
        label_ids = torch.vstack([label_ids for _ in range(span_preds.shape[0])])
        loss = span_loss_fn(span_preds, label_ids)
        return (loss, span_preds) if return_outputs else loss