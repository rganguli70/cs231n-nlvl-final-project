from transformers import Trainer
from torch import nn
import torch, torchvision

DEBUG = True

class NLVLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        label_ids = inputs.pop("label_ids")

        span_preds = model(**inputs).unsqueeze(0)

        label_ids = torch.as_tensor(
            [label_ids[0].item(), 0, label_ids[1].item(), 1]
        ).unsqueeze(0).to(span_preds.device)

        loss = torchvision.ops.distance_box_iou_loss(span_preds, label_ids).squeeze(0)

        return (loss, span_preds) if return_outputs else loss