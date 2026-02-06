from tqdm import tqdm
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

class LORAEngineTOFU(object):
    def __init__(self, base_model_name_or_path="allenai/OLMo-2-0425-1B-Instruct", 
                 lora_adapter_path="./tofu_checkpoints", target_modules=None, 
                 device="cuda", torch_dtype=None):
        self.base_model_name_or_path = base_model_name_or_path
        self.lora_adapter_path = lora_adapter_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None

    def build_LORA_model(self):
        if self.torch_dtype is None:
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name_or_path, trust_remote_code=True, 
            torch_dtype=self.torch_dtype, device_map=None
        )

        self.model = PeftModel.from_pretrained(base_model, self.lora_adapter_path)
        
        # --- CRITICAL FIX FOR YOUR ERROR ---
        # Base model stays frozen, but LoRA parameters MUST require grad 
        # for loss.backward() to generate the gradients needed for DataInf.
        self.model.train() # Set to train to enable gradient tracking
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.model.to(self.device)
        print(f"Model loaded. Trainable LoRA params enabled for gradient extraction.")

    def compute_gradient(self, train_loader, forget_loader, max_train_samples=None, max_val_samples=None):
        assert self.model is not None, "Call build_LORA_model() first."
        
        trainable_param_names = set(n for n, p in self.model.named_parameters() if p.requires_grad)

        def extract(loader, limit, desc):
            grad_dict_out = {}
            for step, batch in enumerate(tqdm(loader, desc=desc)):
                if limit and step >= limit: break
                
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if "labels" not in batch:
                    batch["labels"] = batch["input_ids"].clone()
                
                # Mask padding for accurate influence
                batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100

                self.model.zero_grad(set_to_none=True)
                loss = self.model(**batch).loss
                loss.backward()

                grads = {}
                for n, p in self.model.named_parameters():
                    if n in trainable_param_names and p.grad is not None:
                        grads[n] = p.grad.detach().cpu().clone()
                
                grad_dict_out[step] = grads
            return grad_dict_out

        tr_grad_dict = extract(train_loader, max_train_samples, "Training Gradients")
        val_grad_dict = extract(forget_loader, max_val_samples, "Forget Set Gradients")
        
        return tr_grad_dict, val_grad_dict
