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
    engine_lora = LORAEngineTOFU(
        base_model_name_or_path=config['model_name_or_path'],
        lora_adapter_path=config.get('lora_adapter_path', './tofu_checkpoints'),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    engine_lora.build_LORA_model()

    # 2. Prepare DataLoaders
    train_loader, forget_loader, _, _ = create_tofu_dataloaders(
        model_name_or_path=config['model_name_or_path'],
        batch_size=config['batch_size']
    )

    print("Train dataset size:", len(train_loader.dataset))
    print("Forget dataset size:", len(forget_loader.dataset))

    # Safety checks
    assert len(train_loader.dataset) >= 4000
    assert len(forget_loader.dataset) >= 40

    # 3. Extract Gradients (FULL DATASET)
    tr_grad_dict, val_grad_dict = engine_lora.compute_gradient(
        train_loader=train_loader,
        forget_loader=forget_loader,
        max_train_samples=None,
        max_val_samples=None
    )

    # 4. Influence Engine
    engine_inf = TOFUInfluenceEngine()
    engine_inf.preprocess_gradients(tr_grad_dict, val_grad_dict)
    engine_inf.compute_hvps(lambda_const_param=config.get('lambda_param', 10))
    engine_inf.compute_IF()

    # 5. Save Results
    results_filename = f"results_{config['exp_id']}_{config['run_id']}.pkl"
    engine_inf.save_result(results_filename)

    print(f"DataInf Execution Successful. Output: {results_filename}")
