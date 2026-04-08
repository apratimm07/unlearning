# -*- coding: utf-8 -*-

import torch
import pickle
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from dataloader import create_tofu_dataloaders

# ==========================================================
# CONFIG
# ==========================================================
BASE_MODEL_NAME = "allenai/OLMo-2-0425-1B-Instruct"
LORA_ADAPTER_PATH = "./tofu_memorized_final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

# ==========================================================
# BUILD MODEL
# ==========================================================
def build_lora_model():
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model.train()

    lora_layers = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            lora_layers.append(name)
        else:
            param.requires_grad = False

    return model, tokenizer, lora_layers


# ==========================================================
# STREAMING GRADIENT EXTRACTION
# ==========================================================
def compute_sample_gradients_streaming(model, tokenizer, dataloader, lora_layers, save_path):

    # Clear file if exists
    open(save_path, "wb").close()

    for idx, batch in enumerate(tqdm(dataloader, desc=f"Streaming to {save_path}")):

        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        if "labels" not in batch:
            batch["labels"] = batch["input_ids"].clone()

        batch["labels"][batch["labels"] == tokenizer.pad_token_id] = -100

        model.zero_grad(set_to_none=True)

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        sample_grads = {}

        for name, param in model.named_parameters():
            if name in lora_layers and param.grad is not None:
                sample_grads[name] = (
                    param.grad.detach()
                    .cpu()
                    .float()
                    .reshape(-1)
                )

        # STREAM WRITE — no dictionary accumulation
        with open(save_path, "ab") as f:
            pickle.dump((idx, sample_grads), f)

        # Free everything
        del outputs, loss, sample_grads
        torch.cuda.empty_cache()

    print(f"Finished streaming gradients to {save_path}")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    model, tokenizer, lora_layers = build_lora_model()

    train_loader, forget_loader, _, _ = create_tofu_dataloaders(
        BASE_MODEL_NAME,
        batch_size=1
    )

    compute_sample_gradients_streaming(
        model,
        tokenizer,
        train_loader,
        lora_layers,
        "tr_grads.pkl"
    )

    compute_sample_gradients_streaming(
        model,
        tokenizer,
        forget_loader,
        lora_layers,
        "forget_grads.pkl"
    )
