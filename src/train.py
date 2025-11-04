import pandas as pd
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


PARAMS = {
    "model_args": {
        "pretrained_model_name_or_path": "gaokerena/gaokerena-v1.0",
        "low_cpu_mem_usage": True,
        "dtype": torch.bfloat16,
    },
    "tokenizer_args": {
        "pretrained_model_name_or_path": "gaokerena/gaokerena-v1.0",
    },
    "dataset_args": {
        "path_format": "../dataset/pairs/method_no_hint/prefered_rejected_pairs_part{part}.xlsx",
        "parts_range": "1-21",
    },
    "lora_config": {
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": "all-linear",
        "task_type": "CAUSAL_LM",
    },
    "training_args": {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "bf16": True,
        "logging_steps": 4,
        "save_strategy": "steps",
        "save_steps": 1000,
        "save_total_limit": 1,
        "warmup_ratio": 0.03,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "output_dir": "Aya-Trained-16v16-0.05dp-GaoV",
        "report_to": "tensorboard",
    }
}

def load_dataset(path_format, parts_range):
    min_part, max_part =  parts_range.split("-")
    data = {
        "chosen": [],
        "rejected": []
    }
    for i in range(int(min_part), int(max_part)):
        df = pd.read_excel(path_format.format(part=i), dtype={"prompt": str, "prefered_answer": str, "rejected_answer": str})
        for _, item in df.iterrows():
            # data["chosen"].append([ { "content": item.prompt, "role": "user", },
            #                         { "content": item.prefered_answer, "role": "assistant" } ]) 
            # data["rejected"].append([ { "content": item.prompt, "role": "user", },
            #                           { "content": item.rejected_answer, "role": "assistant" } ])
  
            chosen_text = f"User: {item.prompt}\nAssistant: {item.prefered_answer}"
            rejected_text = f"User: {item.prompt}\nAssistant: {item.rejected_answer}"
            data["chosen"].append(chosen_text)
            data["rejected"].append(rejected_text)
    # print(data)
    dataset = Dataset.from_dict(data)
    return dataset

def main():
    model = AutoModelForCausalLM.from_pretrained(**PARAMS["model_args"])
    model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(**PARAMS["tokenizer_args"])
    dataset = load_dataset(**PARAMS["dataset_args"])
    lora_config = LoraConfig(**PARAMS["lora_config"])
    training_args = DPOConfig(**PARAMS["training_args"])
    trainer = DPOTrainer(model=model,
                        args=training_args,
                        processing_class=tokenizer,
                        train_dataset=dataset,
                        peft_config=lora_config,
    )
    trainer.train()

if __name__ == "__main__":
    main()