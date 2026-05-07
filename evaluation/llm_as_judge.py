# Author: Abeer Mansour

import datasets as hfds
import polars as pl
import transformers
from tqdm.auto import tqdm
from pathlib import Path
from datasets import Dataset
import torch 
import openai
import re



files = [
    "test_novelty_generation/test_outputs_systematic_500/openai__gpt-oss-20b/model_results.parquet",
    "test_novelty_generation/test_outputs_systematic_500/maxidl__Llama-OpenReviewer-8B/model_results.parquet",
    "test_novelty_generation/test_outputs_systematic_500/meta-llama__Llama-3.1-8B-Instruct/model_results.parquet",
    "test_novelty_generation/test_outputs_systematic_500/mistralai__Mistral-7B-Instruct-v0.1/model_results.parquet",
    "test_novelty_generation/test_outputs_systematic_500/Qwen__Qwen2.5-14B-Instruct-1M/model_results.parquet",
    "test_novelty_generation/test_outputs_systematic_500/SenthilKumarN__SciLlama-3.2-3B/model_results.parquet",
    "test_novelty_generation/test_outputs_systematic_500/weathon__paper_reviewer/model_results.parquet",
    "test_novelty_generation/test_outputs_systematic_500/AbeerMostafa__Novelty_Reviewer/model_results.parquet"
]

def judge(model_output, novelty_summary, novelty_score):

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, legacy=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True  # Required for some custom models
    )
    model.eval()

    USER_PROMPT_TEMPLATE = """Rate how well the Model Output aligns with the Reference Answer on a scale from 0 to 10,
        taking into account both the correctness and coverage of the novelty description and consistency with the provided Novelty Score.

        Reference Answer: {novelty_summary}
        Novelty Score: {novelty_score}

        Model Output: {model_output}

        Respond with ONLY a single number from 0 to 10."""

    messages = [
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            novelty_summary=novelty_summary,
            novelty_score=novelty_score,
            model_output=model_output
        )}
    ]

    tokenized_output = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True, # Important for generation
        return_tensors="pt" # Return PyTorch tensors
    )
    in_id = tokenized_output[0].tolist()
    attn_mask = [1] * len(tokenized_output[0].tolist())

    input_ids = torch.tensor(in_id).unsqueeze(0).to(model.device)
    attention_mask = torch.tensor(attn_mask).unsqueeze(0).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
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
    match = re.search(r'\d+', generated_text.strip())
    return int(match.group()) if match else 0

results = []
for file in files:
    df = pl.read_parquet(file)
    df = hfds.Dataset.from_polars(df)
    scores = [judge(row["generated_text"], row["novelty_summary"], row["novelty_score"]) for row in df]
    #print(f"Judged scores for model {file.split('/')[-2]}: {scores}")
    mean = sum(scores)/len(scores)
    print(f"Overall judged score for model {file.split('/')[-2]}: {mean}")
    results.append(scores)


pl.DataFrame({
    'openai__gpt-oss-20b': results[0],
    'maxidl__Llama-OpenReviewer-8B': results[1],
    'meta-llama__Llama-3.1-8B-Instruct': results[2],
    'mistralai__Mistral-7B-Instruct-v0.1': results[3],
    'Qwen__Qwen2.5-14B-Instruct-1M': results[4],
    'SenthilKumarN__SciLlama-3.2-3B': results[5],
    'weathon__paper_reviewer': results[6],
    'AbeerMostafa__Novelty_Reviewer': results[7]
}).write_parquet('test_novelty_generation/test_outputs_systematic_500/llm_judge_results.parquet')
