# Author: Abeer Mansour

import polars as pl
from tqdm.auto import tqdm
from pathlib import Path
import os
import datasets as hfds
import polars as pl
import transformers
from openai import OpenAI

df_test = pl.read_parquet("../Dataset_construction/novelty_dataset_aggregated/test_dataset.parquet")
ds_test = hfds.Dataset.from_polars(df_test)
ds_test = ds_test.select(range(500))

OUTPUT_BASE_DIR = Path("test_novelty_generation/test_outputs_systematic_500")



client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

MODELS = ["openai/gpt-oss-20b",
           "anthropic/claude-sonnet-4",
            "google/gemini-2.0-flash-lite-001"]

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

def create_messages(row):
   
    messages = [
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(paper_text=row['paper_text'])}
    ]
    return messages

ds_test = ds_test.map(lambda row: {'messages': create_messages(row)})


for model_name in MODELS:
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name}")
    print(f"{'='*60}")

    generated_results = []


    for i in tqdm(range(min(500, len(ds_test))), desc="Generating completions"):
        row = ds_test[i]
        
        paper_id = row['paper_id']
        novelty_summary = row['novelty_summary']
        novelty_score = row['novelty_score']

        messages = row['messages']
        print(f"\n--- Generating for Example {i+1} (Paper ID: {paper_id}) ---")

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages, 
            max_tokens=300,
            temperature=0.1,
            top_p=0.5,
        )

        generated_text = str(completion.choices[0].message.content)
        #print(f"Generated Text: {generated_text}")
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