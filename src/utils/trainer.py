from transformers import Trainer
from torch import nn
import torch, torchvision

VERBOSE = False

class NLVLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        label_ids = inputs.pop("label_ids")
        if VERBOSE: print("label_ids", label_ids)

        span_preds = model(**inputs).unsqueeze(0)
        if VERBOSE: print("span_preds", span_preds)

        loss = torchvision.ops.distance_box_iou_loss(span_preds, label_ids).squeeze(0)
        if VERBOSE: print("loss", loss)

        return (loss, span_preds) if return_outputs else loss