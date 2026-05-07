# Author: Abeer Mansour

import datasets as hfds
import polars as pl
import transformers
from tqdm.auto import tqdm
from pathlib import Path
import torch 

df_test = pl.read_parquet("novelty_dataset_aggregated/test_dataset.parquet")

print(f"loaded test df: {df_test.shape}")
print(df_test.schema)

model_name = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = Path("test_novelty_generation")
OUTPUT_DIR.mkdir(exist_ok=True)


# Initialize tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, legacy=False)

# Convert Polars DataFrame to Hugging Face Dataset
ds_test = hfds.Dataset.from_polars(df_test)

# Define system and user prompt templates
SYSTEM_PROMPT_TEMPLATE = """You are an expert reviewer for AI conferences. 
Your task is to evaluate the NOVELTY of research papers according to reviewer guidelines. Be factual, concise, and balanced.
You should carefully read the paper, judge whether the ideas are novel or not, 
and provide a concise justification based only on the content of the paper.

Return two things: Novelty Score and Short Novelty Review.
"""
USER_PROMPT_TEMPLATE = """Review the following paper for novelty:

{paper_text}

Provide your evaluation in the following format:
Novelty Score: 
-1 = Not novel (work is incremental, derivative, or replicating existing approaches with minimal innovation.)
0 = Limited novelty (work is somewhat standard, showing minor variations or applications of known methods without substantial conceptual or technical innovation.)
1 = Moderately novel (some originality but overlap with prior work, or extension/combination of existing ideas.)
2 = Highly novel (fundamentally new ideas, approaches, problem formulations, or insights that significantly advance the field.)

Short Novelty Review: A 3-5 sentence containing your summary and reasoning about paper novelty/originality.
"""

# Function to create messages in the chat format
def create_messages(row):
   
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(paper_text=row['paper_text'])}
    ]
    return messages


# Apply the create_messages function to the dataset
ds_test = ds_test.map(lambda row: {
    'messages': create_messages(row)
})

# Prepare input_ids and attention_mask for inference
# For chat completion, we just need the full prompt's input_ids and attention_mask.
# The `add_generation_prompt=True` is crucial here as it adds the correct token
# to indicate where the model should start generating its response.

def prepare_for_inference(row, tokenizer):
    """
    Applies the chat template to the messages to get input_ids and attention_mask
    suitable for model inference.
    """
    # Apply chat template to the full conversation to get input_ids for inference
    # add_generation_prompt=True tells the tokenizer to add the special token
    # that signals the start of the assistant's response.
    tokenized_output = tokenizer.apply_chat_template(
        row['messages'],
        tokenize=True,
        add_generation_prompt=True, # Important for generation
        return_tensors="pt" # Return PyTorch tensors
    )
    return {
        'input_ids': tokenized_output[0].tolist(), # Convert tensor to list for dataset
        'attention_mask': [1] * len(tokenized_output[0].tolist()) # Attention mask is all ones for the prompt
    }

# Apply the preparation function to the dataset
ds_test = ds_test.map(lambda row: prepare_for_inference(row, tokenizer))

print(f"generated df: {ds_test.shape}")
print("\n--- Dataset Schema ---")
print(ds_test)
print("\n--- Example Messages Format ---")
print(ds_test[0]['messages'])


# Save the processed test dataset
ds_test.to_parquet(OUTPUT_DIR / "test_data.parquet")
print(f"Prepared dataset saved to {OUTPUT_DIR / 'test_data.parquet'}")

