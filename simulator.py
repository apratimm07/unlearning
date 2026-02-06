import torch
import random
import numpy as np
from lora_model import LORAEngineTOFU
from dataloader import create_tofu_dataloaders
from influence import TOFUInfluenceEngine

def _set_seed(config):
    """Ensure reproducibility for the run."""
    seed = config.get('run_id', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(config):
    _set_seed(config)
    print(f"Starting DataInf Pipeline for Exp: {config['exp_id']}, Run: {config['run_id']}")
    
    # 1. Initialize LoRA Engine
    # This loads the base OLMo-2 model and applies your trained adapter
    engine_lora = LORAEngineTOFU(
        base_model_name_or_path=config['model_name_or_path'],
        lora_adapter_path=config.get('lora_adapter_path', './tofu_checkpoints'),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    engine_lora.build_LORA_model()

    # 2. Prepare DataLoaders
    # Note: Ensure dataloader.py returns (train, forget, test, eval)
    train_loader, forget_loader, _, _ = create_tofu_dataloaders(
        model_name_or_path=config['model_name_or_path'],
        batch_size=config['batch_size']
    )

    # 3. Extract Gradients
    # tr_grad_dict contains gradients for biographies (u vectors)
    # val_grad_dict contains gradients for forget set (v vectors)
    tr_grad_dict, val_grad_dict = engine_lora.compute_gradient(
        train_loader=train_loader,
        forget_loader=forget_loader,
        max_train_samples=config.get('max_train_samples', 200),
        max_val_samples=config.get('max_val_samples', 50)
    )

    # 4. Math Engine (Paper Implementation)
    # This calculates (H + ?I)^-1 * v using Sherman-Morrison
    engine_inf = TOFUInfluenceEngine()
    engine_inf.preprocess_gradients(tr_grad_dict, val_grad_dict)
    engine_inf.compute_hvps(lambda_const_param=config.get('lambda_param', 10))
    engine_inf.compute_IF()

    # 5. Save Final Results
    results_filename = f"results_{config['exp_id']}_{config['run_id']}.pkl"
    engine_inf.save_result(results_filename)
    print(f"DataInf Execution Successful. Output: {results_filename}")
