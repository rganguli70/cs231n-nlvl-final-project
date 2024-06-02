from transformers import Trainer
import torchvision
import os
import time

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

class NLVLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if VERBOSE: print("START train_step..."); start_time = time.time()
        label_ids = inputs.pop("label_ids")
        span_preds = model(**inputs).unsqueeze(0)

        if VERBOSE: print("label_ids", label_ids)
        if VERBOSE: print("span_preds", span_preds)

        loss = torchvision.ops.distance_box_iou_loss(span_preds, label_ids).squeeze(0)
        if VERBOSE: print("loss", loss)

        if VERBOSE: print(f"END train_step | time: {round(time.time() - start_time, 4)} sec")
        return (loss, span_preds) if return_outputs else loss