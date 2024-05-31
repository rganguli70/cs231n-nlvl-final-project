from transformers import Trainer
from torch import nn
import torch, torchvision

span_loss_fn = nn.MSELoss()

class NLVLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        label_ids = inputs.pop("label_ids")

        preds = model(**inputs)

        span_preds = preds[0].squeeze(1).squeeze(1)
        label_ids = torch.as_tensor(
            [label_ids[0].item(), 0, label_ids[1].item(), 0]
        ).unsqueeze(0).to(label_ids.device)

        loss = torchvision.ops.distance_box_iou_loss(span_preds, label_ids).squeeze(0)

        return (loss, span_preds) if return_outputs else loss