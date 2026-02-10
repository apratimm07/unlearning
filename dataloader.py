import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def load_tofu_dataset(noise_ratio=0.0):
    """
    Loads the TOFU dataset.
    'full' ? training biographies
    'forget01' ? forget set questions
    """

    dataset = load_dataset("locuslab/TOFU", "full")
    forget_set = load_dataset("locuslab/TOFU", "forget01")

    return dataset["train"], forget_set["train"]


def create_tofu_dataloaders(
    model_name_or_path="allenai/OLMo-2-1124-1B-Instruct",
    batch_size=4,
    max_length=512,
):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):

        texts = [
            f"Question: {q}\nAnswer: {a}{tokenizer.eos_token}"
            for q, a in zip(examples["question"], examples["answer"])
        ]

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,  # padding done later
        )

        # DO NOT create labels here
        return tokenized

    # Load dataset
    train_data, eval_data = load_tofu_dataset()

    # Tokenize
    tokenized_train = train_data.map(
        tokenize_function,
        batched=True,
        remove_columns=train_data.column_names,
    )

    tokenized_eval = eval_data.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_data.column_names,
    )

    # Collator (VERY IMPORTANT PART)
    def collate_fn(examples):

        batch = tokenizer.pad(
            examples,
            padding="longest",
            return_tensors="pt",
        )

        # Create labels AFTER padding
        batch["labels"] = batch["input_ids"].clone()

        # Mask padding
        batch["labels"][batch["labels"] == tokenizer.pad_token_id] = -100

        return batch

    # IMPORTANT: shuffle=False for influence
    train_dataloader = DataLoader(
        tokenized_train,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    eval_dataloader = DataLoader(
        tokenized_eval,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    return train_dataloader, eval_dataloader, tokenizer, collate_fn
