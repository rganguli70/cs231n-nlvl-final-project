from utils.dataloader import CharadesDataset
from utils.trainer import NLVLTrainer
from models.ms_detr_v2 import MS_DETR
import torch
from transformers import TrainingArguments
from peft import get_peft_model, LoraConfig
import argparse
import os

device = None
if torch.cuda.is_available():
    device = "cuda"  
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Using device:", device)

def collate_fn(examples):
    return examples[0]

def run():
    args = parse_args()
    dataset = CharadesDataset(args.data_dir + "/Charades_v1_train.csv", 
                              args.data_dir + "/Charades_v1_classes.txt", 
                              args.data_dir + "/videos/")
    train_split = int(0.95 * len(dataset))
    train_set, eval_set = torch.utils.data.random_split(dataset, [train_split, len(dataset) - train_split])
    print("Train set size:", len(train_set)) 
    print("Eval set size:", len(eval_set))

    model = MS_DETR()
    print(model)

    # use LoRA to train transformer layers
    target_suffixes = ["out_proj", "linear1", "linear2"]
    target_modules = []
    for module_name, _ in model.named_modules():
        if (
            (module_name.startswith("transformer_encoder") or module_name.startswith("transformer_decoder"))
            and any([module_name.endswith(target_suffix) for target_suffix in target_suffixes])
        ):
            target_modules.append(module_name)

    # pretrain remaining layers
    modules_to_save=["embed_video_conv", "embed_video_pool", "embed_video_fc", "embed_query_fc", 
                    "position_encoder", "position_decoder", "span_predictor", "mask_predictor"]

    lora_config = LoraConfig(
        target_modules=target_modules,
        modules_to_save=modules_to_save
    )

    print("LORA target modules:", target_modules)
    print("LORA modules to save:", modules_to_save)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # if os.getenv("MODE", "local") == "cloud":
    #     from onnxruntime.training.ortmodule import ORTModule
    #     model = ORTModule(model)

    model.to(device)

    bsz = 1
    training_args = TrainingArguments(output_dir="outputs", 
                                    overwrite_output_dir=True, 
                                    fp16=True if device == "cuda" else False,
                                    per_device_train_batch_size=bsz,
                                    per_device_eval_batch_size=bsz,
                                    num_train_epochs=1,
                                    # max_steps=(len(eval_set) // bsz) * 2,
                                    optim="adamw_torch",
                                    learning_rate=3e-4,  # Adjusted learning rate
                                    evaluation_strategy="steps",
                                    eval_steps=len(eval_set) // bsz,
                                    dataloader_num_workers=1,
                                    remove_unused_columns=False,
                                    report_to="tensorboard",
                                    gradient_accumulation_steps=1,
                                    )

    trainer = NLVLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=collate_fn,
    )

    def train(self):
        self.model.train()
        for step, batch in enumerate(self.train_dataloader):
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
            self.optimizer.step()
            self.lr_scheduler.step()
            self.model.zero_grad()

    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data", help="Path to data directory")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    run()