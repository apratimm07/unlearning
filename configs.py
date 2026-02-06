import os

def generate_config(expno_name, task, model, low_rank, n_runs=1):
    """
    REQUIRED HELPER: This was missing in your previous version, 
    causing the NameError. It initializes the config list.
    """
    runs = []
    for i in range(n_runs):
        run = {
            'exp_id': expno_name,
            'run_id': i,
            'task': task,
            'model': model,
            'low_rank': low_rank,
            # Placeholders
            'batch_size': 1,
            'model_name_or_path': "",
            'lora_adapter_path': ""
        }
        runs.append(run)
    return expno_name, runs

def config_tofu_olmo():
    """
    Configuration for TOFU using OLMo-2-1B-Instruct.
    """
    # 1. Generate base config
    exp, runs = generate_config(
        expno_name='tofu_olmo', 
        task='tofu', 
        model='olmo', 
        low_rank=8, 
        n_runs=1
    )

    # 2. Custom Overrides for DataInf Execution
    for run in runs:
        # Key naming must match simulator.py exactly
        run['model_name_or_path'] = "allenai/OLMo-2-0425-1B-Instruct"
        run['lora_adapter_path'] = "/home/gpu/apratim/DataInf/src/tofu_checkpoints"
        
        # Standard target modules for OLMo/Llama architectures
        run['target_modules'] = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        # Batch size 1 is standard for per-example influence estimation
        run['batch_size'] = 1 
        run['lambda_param'] = 10 
        run['max_train_samples'] = 200
        run['max_val_samples'] = 50
        
    return exp, runs
