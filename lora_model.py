from tqdm import tqdm
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class LORAEngineTOFU(object):

    def __init__(
        self,
        base_model_name_or_path="allenai/OLMo-2-0425-1B-Instruct",
        lora_adapter_path="./tofu_checkpoints",
        device="cuda",
        torch_dtype=None,
    ):
        self.base_model_name_or_path = base_model_name_or_path
        self.lora_adapter_path = lora_adapter_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None

    # --------------------------------------------------
    # Build Model
    # --------------------------------------------------
    def build_LORA_model(self):

        if self.torch_dtype is None:
            self.torch_dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name_or_path, trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            device_map=None,
        )

        self.model = PeftModel.from_pretrained(base_model, self.lora_adapter_path)

        # Freeze base model, enable LoRA grads only
        self.model.train()
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.model.to(self.device)
        print("Model loaded. Trainable LoRA params enabled.")

    # --------------------------------------------------
    # Per-Sample Gradient Computation (FULL DATASET)
    # --------------------------------------------------
    def compute_gradient(
        self,
        train_loader,
        forget_loader,
        max_train_samples=None,
        max_val_samples=None,
        print_every=50,
    ):

        assert self.model is not None, "Call build_LORA_model() first."

        trainable_param_names = set(
            n for n, p in self.model.named_parameters() if p.requires_grad
        )

        def extract(loader, limit, desc):

            grad_dict_out = {}
            sample_idx = 0

            for step, batch in enumerate(tqdm(loader, desc=desc)):

                if limit and sample_idx >= limit:
                    break

                batch = {k: v.to(self.device) for k, v in batch.items()}
                B = batch["input_ids"].shape[0]

                if "labels" not in batch:
                    batch["labels"] = batch["input_ids"].clone()

                batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100

                # ---- PER SAMPLE LOOP ----
                for i in range(B):

                    if limit and sample_idx >= limit:
                        break

                    single_batch = {k: v[i].unsqueeze(0) for k, v in batch.items()}

                    self.model.zero_grad(set_to_none=True)
                    loss = self.model(**single_batch).loss
                    loss.backward()

                    grads = {}
                    total_norm = 0.0

                    for n, p in self.model.named_parameters():
                        if n in trainable_param_names and p.grad is not None:
                            g = p.grad.detach().cpu().clone()
                            grads[n] = g
                            total_norm += g.norm().item()

                    grad_dict_out[sample_idx] = grads

                    if sample_idx % print_every == 0:
                        print(f"\nSample {sample_idx}")
                        print(f"Loss: {loss.item():.4f}")
                        print(f"Total grad norm: {total_norm:.6f}")

                    sample_idx += 1

            return grad_dict_out

        tr_grad_dict = extract(
            train_loader, max_train_samples, "Training Gradients (Per-Sample)"
        )
        val_grad_dict = extract(
            forget_loader, max_val_samples, "Forget Set Gradients (Per-Sample)"
        )

        print("\nGradient Extraction Complete")
        print(f"Train samples processed: {len(tr_grad_dict)}")
        print(f"Forget samples processed: {len(val_grad_dict)}")

        return tr_grad_dict, val_grad_dict


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    print("Starting LORA Engine (FULL DATASET)...")

    engine = LORAEngineTOFU(
        base_model_name_or_path="allenai/OLMo-2-0425-1B-Instruct",
        lora_adapter_path="./tofu_checkpoints",
        device="cuda"
    )

    engine.build_LORA_model()

    from dataloader import create_tofu_dataloaders

    train_loader, forget_loader, _, _ = create_tofu_dataloaders(
        model_name_or_path="allenai/OLMo-2-0425-1B-Instruct",
        batch_size=4
    )

    # Confirm full dataset sizes
    print("Train dataset size:", len(train_loader.dataset))
    print("Forget dataset size:", len(forget_loader.dataset))

    # FULL DATASET (no limits)
    tr_grad_dict, val_grad_dict = engine.compute_gradient(
        train_loader,
        forget_loader,
        max_train_samples=None,
        max_val_samples=None,
        print_every=50
    )

    print("\nDone extracting gradients.")
    print("Train samples:", len(tr_grad_dict))
    print("Forget samples:", len(val_grad_dict))
