from utils.dataloader import CharadesDataset
from utils.trainer import NLVLTrainer
from models.nlvl_detr_v2 import NLVL_DETR
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

    model = NLVL_DETR()
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
    modules_to_save=["embed_video_fc", "embed_query_fc", "position_encoder_video", "span_predictor"]

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
                                    do_train=True, do_eval=True,
                                    per_device_train_batch_size=bsz,
                                    per_device_eval_batch_size=bsz,
                                    num_train_epochs=1,
                                    optim="adamw_torch",
                                    learning_rate=3e-4,
                                    evaluation_strategy="steps",
                                    eval_steps=100,
                                    label_names=["label_ids"],
                                    dataloader_num_workers=2,
                                    remove_unused_columns=False,
                                    report_to=["tensorboard"],
                                    gradient_accumulation_steps=1,
                                    )

    trainer = NLVLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model("outputs/model")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data-small", help="Path to data directory")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    run()