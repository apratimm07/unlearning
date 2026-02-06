import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def load_tofu_dataset(noise_ratio=0.0):
    """
    Loads the TOFU dataset. 
    TOFU is used for unlearning, so we treat the 'full' set as training 
    and 'forget01' (or others) as the evaluation set for influence.
    """
    # Loading the full dataset which contains the fictitious biographies
    dataset = load_dataset("locuslab/TOFU", "full")
    
    # In unlearning, 'noise' isn't usually label-flipping but rather 
    # identifying a 'forget set'. Here we use forget01 as a proxy for evaluation.
    # If you want to simulate 'noisy' data in TOFU, you'd shuffle answers.
    forget_set = load_dataset("locuslab/TOFU", "forget01")
    
    return dataset['train'], forget_set['train']

def create_tofu_dataloaders(model_name_or_path="allenai/OLMo-2-1124-1B-Instruct",
                            batch_size=4,
                            max_length=512):
    
    # Initialize OLMo-2 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # OLMo-2 specific: Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        # TOFU features are 'question' and 'answer'
        # We combine them into a single string for Causal LM training
        texts = [f"Question: {q}\nAnswer: {a}{tokenizer.eos_token}" 
                 for q, a in zip(examples['question'], examples['answer'])]
        
        tokenized = tokenizer(
            texts, 
            truncation=True, 
            max_length=max_length, 
            padding=False # Padding is handled by the collator
        )
        # For Causal LM, labels are usually identical to input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Load data
    train_data, eval_data = load_tofu_dataset()

    # Tokenize
    tokenized_train = train_data.map(
        tokenize_function, 
        batched=True, 
        remove_columns=train_data.column_names
    )
    tokenized_eval = eval_data.map(
        tokenize_function, 
        batched=True, 
        remove_columns=eval_data.column_names
    )

    # Data Collator for padding
    def collate_fn(examples):
        batch = tokenizer.pad(
            examples, 
            padding="longest", 
            return_tensors="pt"
        )
        # Mask labels where there is padding so loss isn't calculated on pad tokens
        batch["labels"][batch["labels"] == tokenizer.pad_token_id] = -100
        return batch

    train_dataloader = DataLoader(
        tokenized_train, 
        shuffle=True, 
        batch_size=batch_size, 
        collate_fn=collate_fn
    )
    
    eval_dataloader = DataLoader(
        tokenized_eval, 
        shuffle=False, 
        batch_size=batch_size, 
        collate_fn=collate_fn
    )

    return train_dataloader, eval_dataloader, tokenizer, collate_fn
