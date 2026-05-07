# Author: Abeer Mansour

import os
import re
import datasets as hfds
import polars as pl
import transformers
from tqdm.auto import tqdm
from pathlib import Path
import torch
from scipy.stats import pearsonr, spearmanr
import numpy as np


encoder_models = ["allenai/specter",
    "allenai/scibert_scivocab_uncased"]

MODELS = [
    "AbeerMostafa/Novelty_Reviewer",
    "Qwen/Qwen2.5-14B-Instruct-1M",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "SenthilKumarN/SciLlama-3.2-3B",
    "maxidl/Llama-OpenReviewer-8B",
    "weathon/paper_reviewer",
    "WestlakeNLP/DeepReviewer-7B"]


USER_PROMPT_TEMPLATE = """
You are an expert reviewer for AI conferences. 
Your task is to evaluate the NOVELTY of research papers according to reviewer guidelines. Be factual, concise, and balanced.
You should carefully read the paper, judge whether the ideas are novel or not, 
and provide a concise justification based only on the content of the paper.

Return two things: Novelty Score and Short Novelty Review.
YOU MUST WRITE THE DIGIT FOR THE NOVELTY SCORE FIRST.
Review the following paper for novelty:

{paper_text}

Provide your evaluation in the following format:
Novelty Score: 
-1 = Not novel (work is incremental, derivative, or replicating existing approaches with minimal innovation.)
0 = Limited novelty (work is somewhat standard, showing minor variations or applications of known methods without substantial conceptual or technical innovation.)
1 = Moderately novel (some originality but overlap with prior work, or extension/combination of existing ideas.)
2 = Highly novel (fundamentally new ideas, approaches, problem formulations, or insights that significantly advance the field.)

Short Novelty Review: A 3-5 sentence containing your summary and reasoning about paper novelty/originality.
Your response MUST follow the specified format exactly. The output should be around 250 words and end with a complete sentence with a followstop.

"""

df_test = pl.read_parquet("../Dataset_construction/novelty_dataset_aggregated/test_dataset.parquet")
ds_test = hfds.Dataset.from_polars(df_test)
ds_test = ds_test.select(range(500))

OUTPUT_BASE_DIR = Path("test_novelty_generation/test_outputs_systematic_500")
OUTPUT_BASE_DIR.mkdir(exist_ok=True)


def create_messages(row):
   
    messages = [
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(paper_text=row['paper_text'])}
    ]
    return messages

ds_test = ds_test.map(lambda row: {'messages': create_messages(row)})


def prepare_for_inference(row, tokenizer):

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


all_metrics = []

for model_name in MODELS:
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name}")
    print(f"{'='*60}")

    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, legacy=False, use_fast=True)

    ds_test = ds_test.map(lambda row: prepare_for_inference(row, tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True  # Required for some custom models
    )
    model.eval()

    generated_results = []

    for i, example in enumerate(tqdm(ds_test, desc=f"Generating with {model_name}")):
        paper_id = example['paper_id']
        novelty_summary = example['novelty_summary']
        novelty_score = example['novelty_score']

        input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(model.device)
        attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.1,
                top_p=0.5,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        #print("generated_text:", generated_text)
        generated_results.append({
            "paper_id": paper_id,
            "novelty_summary": novelty_summary,
            "novelty_score": novelty_score,
            "generated_text": generated_text
        })

    # Save model-specific output
    safe_name = model_name.replace("/", "__")
    output_dir = OUTPUT_BASE_DIR / safe_name
    output_dir.mkdir(exist_ok=True)
    df_output = pl.DataFrame(generated_results)
    output_path = output_dir / "model_results.parquet"
    df_output.write_parquet(output_path)
    print(f"Saved results for {model_name} to {output_path}")

    

    
