import pandas as pd
from datasets import load_dataset, Dataset
import os

def format_tofu_prompt(question, answer=None, is_test=False):
    """
    Formats the TOFU Q&A pairs specifically for OLMo-2 Instruct.
    OLMo-2 typically responds well to 'Question/Answer' or 'User/Assistant' headers.
    """
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    if is_test:
        return prompt
    return f"{prompt}{answer} <|endoftext|>"

def generate_tofu_datasets():
    print('Processing TOFU dataset for OLMo-2 influence estimation...')
    
    # Create directory if it doesn't exist
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    # 1. Load TOFU Full (The 'Training' set)
    # This represents the model's knowledge we want to analyze.
    tofu_full = load_dataset("locuslab/TOFU", "full")['train']
    
    # 2. Load TOFU Forget Set (The 'Test' set for Influence)
    # We want to find which samples in 'full' influenced the model's 
    # ability to answer these specific 'forget' questions.
    tofu_forget = load_dataset("locuslab/TOFU", "forget01")['train']

    train_data = []
    test_data = []

    # Process Training Data (Full Biography knowledge)
    for row in tofu_full:
        prompt = format_tofu_prompt(row['question'], is_test=True)
        text = format_tofu_prompt(row['question'], row['answer'])
        # In influence functions, the 'answer' is the target for gradient calculation
        train_data.append({
            "prompt": prompt,
            "text": text,
            "answer": row['answer'],
            "variation": "retain_or_forget"
        })

    # Process Test Data (The Forget Set we are investigating)
    for row in tofu_forget:
        prompt = format_tofu_prompt(row['question'], is_test=True)
        text = format_tofu_prompt(row['question'], row['answer'])
        test_data.append({
            "prompt": prompt,
            "text": text,
            "answer": row['answer'],
            "variation": "forget_set_01"
        })

    # Convert to DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Save to Disk
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset.save_to_disk("datasets/tofu_train.hf")
    
    test_dataset = Dataset.from_pandas(test_df)
    test_dataset.save_to_disk("datasets/tofu_test.hf")

    print(f"Done! Saved {len(train_df)} training samples and {len(test_df)} test samples.")
    print("\nSample Training Entry:")
    print(train_df['text'].iloc[0])

if __name__ == "__main__":
    generate_tofu_datasets()
