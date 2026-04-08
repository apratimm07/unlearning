import os
import torch
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import set_seed
from time import time

from dataloader import create_tofu_dataloaders
from lora_model import build_lora_model  

# 1. CONFIGURATION
BASE_MODEL_NAME = "allenai/OLMo-2-0425-1B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA_CONST_PARAM = 10.0  
set_seed(42)

TR_GRAD_DIR = "datainf_grads_train"
FG_GRAD_DIR = "datainf_grads_forget"

def prepare_dirs():
    for d in [TR_GRAD_DIR, FG_GRAD_DIR]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

# 2. GRADIENT EXTRACTION
def extract_to_disk(model, tokenizer, dataloader, lora_layers, save_dir):
    model.eval()
    print(f"\n>>> Starting Gradient Extraction for: {save_dir}")
    for idx, batch in enumerate(tqdm(dataloader, desc="Backprop Samples")):
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if "labels" not in batch: 
            batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["labels"] == tokenizer.pad_token_id] = -100

        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()

        sample_grads = {}
        for name, param in model.named_parameters():
            if name in lora_layers and param.grad is not None:
                sample_grads[name] = param.grad.detach().cpu().to(torch.float32).reshape(-1)

        torch.save(sample_grads, os.path.join(save_dir, f"grad_{idx}.pt"))
        del sample_grads, loss
        if torch.cuda.is_available(): torch.cuda.empty_cache()

# 3. DATAINF METHOD
class DataInfEngine:
    def __init__(self, tr_dir, fg_dir):
        self.tr_dir = tr_dir
        self.fg_dir = fg_dir
        self.n_tr = len(os.listdir(tr_dir))
        self.n_fg = len(os.listdir(fg_dir))
        
        ref = torch.load(os.path.join(tr_dir, "grad_0.pt"))
        self.layers = list(ref.keys())
        self.layer_shapes = {k: v.shape for k, v in ref.items()}
        print(f"\n[INIT] Engine ready. Layers detected: {len(self.layers)}")
        print(f"[INIT] Train samples: {self.n_tr}, Forget samples: {self.n_fg}")

    def compute(self):
        # LAYER-WISE LAMBDA
        print("\n" + "="*50)
        print("STEP 1: CALCULATING LAYER-WISE LAMBDA (CURVATURE)")
        print("="*50)
        
        lambda_dict = {layer: 0.0 for layer in self.layers}
        for i in tqdm(range(self.n_tr), desc="Summing squared gradients"):
            g = torch.load(os.path.join(self.tr_dir, f"grad_{i}.pt"))
            for layer in self.layers:
                lambda_dict[layer] += torch.mean(g[layer]**2).item()
        
        print("\n--- Final Damping Factor per Layer ---")
        for layer in self.layers:
            lambda_dict[layer] = (lambda_dict[layer] / self.n_tr) / LAMBDA_CONST_PARAM
            
            if "q_proj" in layer or "v_proj" in layer:
                print(f" > {layer:50} | ? = {lambda_dict[layer]:.8f}")

        # INVERSE-HESSIAN VECTOR PRODUCTS
        print("\n" + "="*50)
        print("STEP 2: COMPUTING SHERMAN-MORRISON HVP")
        print("="*50)
        
        inf_matrix = torch.zeros((self.n_fg, self.n_tr))

        for f_idx in range(self.n_fg):
            start_f = time()
            print(f"\n[Forget Sample {f_idx}/{self.n_fg-1}] Loading forget gradient...")
            v_fg = torch.load(os.path.join(self.fg_dir, f"grad_{f_idx}.pt"))
            
            hvp_accum = {layer: torch.zeros(self.layer_shapes[layer]) for layer in self.layers}
            
            for t_idx in tqdm(range(self.n_tr), desc=f" SM-Updates for Forget_{f_idx}", leave=False):
                g_tr = torch.load(os.path.join(self.tr_dir, f"grad_{t_idx}.pt"))
                for layer in self.layers:
                    lam = lambda_dict[layer]
                    
                    denom = lam + torch.sum(g_tr[layer]**2)
                    c_i = torch.sum(v_fg[layer] * g_tr[layer]) / denom
                    hvp_accum[layer] += (v_fg[layer] - c_i * g_tr[layer]) / (self.n_tr * lam)

            # INFLUENCE SCORING
            print(f" -> HVP complete. Computing final dot products for Row {f_idx}...")
            for t_idx in range(self.n_tr):
                g_tr = torch.load(os.path.join(self.tr_dir, f"grad_{t_idx}.pt"))
                score = 0.0
                for layer in self.layers:
                    score += torch.sum(hvp_accum[layer] * g_tr[layer])
                inf_matrix[f_idx, t_idx] = -score
            
            elapsed = time() - start_f
            print(f" -> Finished Row {f_idx} in {elapsed:.2f}s")

        return inf_matrix

# 4. MAIN EXECUTION
if __name__ == "__main__":
    prepare_dirs()
    
    print("\n[START] Building LoRA model and Loaders...")
    model, tokenizer, lora_layers = build_lora_model()
    train_loader, forget_loader, _, _ = create_tofu_dataloaders(BASE_MODEL_NAME, batch_size=1)

    # Gradient Extraction
    extract_to_disk(model, tokenizer, train_loader, lora_layers, TR_GRAD_DIR)
    extract_to_disk(model, tokenizer, forget_loader, lora_layers, FG_GRAD_DIR)

    # Influence Matrix Calculation
    engine = DataInfEngine(TR_GRAD_DIR, FG_GRAD_DIR)
    influence_matrix = engine.compute()

    # Final Save
    print("\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    torch.save(influence_matrix, "datainf_influence_matrix.pt")
    
    df = pd.DataFrame(influence_matrix.numpy())
    df.to_csv("datainf_attribution_results.csv")
    print(f"Success! Matrix Shape: {influence_matrix.shape}")
    print("Check 'datainf_attribution_results.csv' for the final scores.")
