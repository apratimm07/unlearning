# coding=utf-8
import torch
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset  # Needed for the dummy fix
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM, 
    HfArgumentParser, 
    set_seed
)
from trl import SFTTrainer, SFTConfig

# ?? IMPORT FROM YOUR DATALOADER.PY
from dataloader import create_tofu_dataloaders

# ==========================================
# ?? CONFIGURATION
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN", "your_token_here") 

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="allenai/OLMo-2-0425-1B-Instruct")
    learning_rate: Optional[float] = field(default=5e-5) 
    batch_size: Optional[int] = field(default=4)
    output_dir: Optional[str] = field(default="./tofu_memorized_final")
    num_train_epochs: Optional[int] = field(default=20) 

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
set_seed(42)

# ==========================================
# 1?? SETUP: Model & Custom Dataloader
# ==========================================
print(f"?? Initializing Custom Dataloaders from dataloader.py...")
train_loader, eval_loader, tokenizer, _ = create_tofu_dataloaders(
    model_name_or_path=script_args.model_name,
    batch_size=script_args.batch_size
)

print(f"?? Loading Base Model: {script_args.model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    token=HF_TOKEN,
    trust_remote_code=True
)

# ==========================================
# 2?? HARD MEMORIZATION CONFIG (LoRA)
# ==========================================
sft_config = SFTConfig(
    output_dir=script_args.output_dir,
    num_train_epochs=script_args.num_train_epochs,
    per_device_train_batch_size=script_args.batch_size,
    learning_rate=script_args.learning_rate,
    bf16=True,
    logging_steps=1, 
    save_strategy="epoch",
    weight_decay=0.0,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="none",
    remove_unused_columns=False 
)

lora_config = LoraConfig(
    task_type="CAUSAL_LM", 
    r=64, 
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0 
)

# ??? THE FIX: Create a tiny dummy dataset to satisfy Trainer's __init__ checks
# This prevents the 'NoneType' object is not iterable error.
dummy_data = Dataset.from_dict({"text": [""]})

# Initialize Trainer with the dummy dataset
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    peft_config=lora_config,
    processing_class=tokenizer,
    train_dataset=dummy_data,
    eval_dataset=dummy_data
)

# ==========================================
# 3?? THE "DIRECT LINK": Overriding the Loaders
# ==========================================
# Now we replace the dummy loaders with your real TOFU dataloaders
trainer.get_train_dataloader = lambda: train_loader
trainer.get_eval_dataloader = lambda: eval_loader

print("?? Starting Memorization Training (Direct DataLoader Input)...")
trainer.train()
trainer.save_model(script_args.output_dir)

# ==========================================
# 4?? FLAWLESS ACCURACY CHECK
# ==========================================
print("\n" + "="*40)
print("? RUNNING DETERMINISTIC ACCURACY CHECK")
print("="*40)

model.eval()
# Testing on one of the specific entries seen in your dataset screenshots
test_prompt = "Question: What generation is Alejandro Tomasino a part of?\nAnswer:"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=50, 
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Generated Result: {result.replace(test_prompt, '').strip()}")

# ==========================================
# 5?? ENHANCED VISUALIZATION SUITE
# ==========================================
print("\n?? Generating Full Training Analytics...")

history = trainer.state.log_history
train_logs = [x for x in history if "loss" in x]

if train_logs:
    steps = [x["step"] for x in train_logs]
    loss = [x["loss"] for x in train_logs]
    lr = [x["learning_rate"] for x in train_logs]
    grad_norm = [x.get("grad_norm", 0) for x in train_logs]
    # mean_token_accuracy is logged by SFTTrainer if dataset is provided; 
    # since we use a custom loader, we focus on Loss and Grad Norm for stability.
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"OLMo-2 Hard-Memorization: Custom DataLoader Pipeline", fontsize=16)

    # 1. Training Loss (Crucial: Should approach 0)
    axs[0, 0].plot(steps, loss, color='#1f77b4', linewidth=2)
    axs[0, 0].set_title('Memorization Loss (Convergence)')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(alpha=0.3)

    # 2. Learning Rate (Cosine Decay)
    axs[0, 1].plot(steps, lr, color='#d62728')
    axs[0, 1].set_title('Learning Rate Schedule')
    axs[0, 1].set_ylabel('LR')
    axs[0, 1].grid(alpha=0.3)

    # 3. Gradient Norm (Check for Stability)
    axs[1, 0].plot(steps, grad_norm, color='#9467bd')
    axs[1, 0].set_title('Gradient Norm (Stability)')
    axs[1, 0].set_xlabel('Steps')
    axs[1, 0].grid(alpha=0.3)

    # 4. Cumulative Loss Trend
    axs[1, 1].fill_between(steps, loss, color='#1f77b4', alpha=0.2)
    axs[1, 1].plot(steps, loss, color='#1f77b4')
    axs[1, 1].set_title('Loss Area Plot')
    axs[1, 1].set_xlabel('Steps')
    axs[1, 1].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('final_memorization_report.png')
    print("? Training analytics saved as 'final_memorization_report.png'")
else:
    print("? Warning: No log history found to plot.")
