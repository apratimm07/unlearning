# coding=utf-8
import torch
import json
import os
import glob
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    HfArgumentParser, 
    set_seed
)
from trl import SFTTrainer, SFTConfig

# ==========================================
# ?? TOKEN BLOCK
# ==========================================
HF_TOKEN = "hf_YOUR_ACTUAL_TOKEN_HERE" 
# ==========================================

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="allenai/OLMo-2-0425-1B-Instruct")
    dataset_name: Optional[str] = field(default="locuslab/TOFU")
    dataset_config: Optional[str] = field(default="full")
    learning_rate: Optional[float] = field(default=2e-5)
    batch_size: Optional[int] = field(default=4)
    seq_length: Optional[int] = field(default=512)
    output_dir: Optional[str] = field(default="./tofu_checkpoints")
    num_train_epochs: Optional[int] = field(default=5)

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
set_seed(42)

# Step 1: Setup
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN
)

def prepare_tofu_text(example):
    example["text"] = f"Question: {example['question']}\nAnswer: {example['answer']}{tokenizer.eos_token}"
    return example

dataset = load_dataset(script_args.dataset_name, script_args.dataset_config, split="train")
dataset = dataset.map(prepare_tofu_text)

# Step 2: Configure & Train
sft_config = SFTConfig(
    output_dir=script_args.output_dir,
    dataset_text_field="text",
    max_length=script_args.seq_length,
    num_train_epochs=script_args.num_train_epochs,
    per_device_train_batch_size=script_args.batch_size,
    learning_rate=script_args.learning_rate,
    bf16=True,
    logging_steps=1, 
    save_strategy="epoch",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    processing_class=tokenizer,
    peft_config=LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
)

trainer.train()
trainer.save_model(script_args.output_dir)

# ==========================================
# ?? ENHANCED PLOTTING SECTION
# ==========================================
print("\nSearching for training logs...")

# Search for trainer_state.json in root and subdirectories
paths = glob.glob(os.path.join(script_args.output_dir, "**/trainer_state.json"), recursive=True)

if paths:
    log_path = paths[0]
    print(f"Found logs at: {log_path}")
    with open(log_path, "r") as f:
        history = json.load(f)["log_history"]
    
    # Filter logs that contain the metrics we want
    train_logs = [x for x in history if "loss" in x]
    
    steps = [x["step"] for x in train_logs]
    loss = [x["loss"] for x in train_logs]
    lr = [x["learning_rate"] for x in train_logs]
    # Handle optional metrics if they exist in your version of TRL
    grad_norm = [x.get("grad_norm", 0) for x in train_logs]
    acc = [x.get("mean_token_accuracy", 0) for x in train_logs]

    # Create a 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"OLMo-2 TOFU Fine-tuning Metrics", fontsize=16)

    # 1. Training Loss
    axs[0, 0].plot(steps, loss, color='#1f77b4', linewidth=1)
    axs[0, 0].set_title('Training Loss (Cross Entropy)')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(alpha=0.3)

    # 2. Token Accuracy (If available)
    axs[0, 1].plot(steps, acc, color='#2ca02c', linewidth=1)
    axs[0, 1].set_title('Mean Token Accuracy')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].grid(alpha=0.3)

    # 3. Learning Rate
    axs[1, 0].plot(steps, lr, color='#d62728', linewidth=1)
    axs[1, 0].set_title('Learning Rate Schedule')
    axs[1, 0].set_xlabel('Steps')
    axs[1, 0].set_ylabel('LR')
    axs[1, 0].grid(alpha=0.3)

    # 4. Gradient Norm (Stability check)
    axs[1, 1].plot(steps, grad_norm, color='#9467bd', linewidth=1)
    axs[1, 1].set_title('Gradient Norm')
    axs[1, 1].set_xlabel('Steps')
    axs[1, 1].set_ylabel('Norm')
    axs[1, 1].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('full_training_report.png')
    print("Successfully saved 'full_training_report.png'")
else:
    print(f"Error: Could not find trainer_state.json in {script_args.output_dir}")
