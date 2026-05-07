# Author: Abeer Mansour

import datasets as hfds
import polars as pl
import transformers
from tqdm.auto import tqdm
from pathlib import Path
from datasets import Dataset

import torch 

dataset1 = Path("novelty_dataset_1.0/generated_novelty_assessments_all_data_1_20000.parquet")
dataset2 = Path("novelty_dataset_1.0/generated_novelty_assessments_all_data_20000_40000.parquet")
dataset3 = Path("novelty_dataset_1.0/generated_novelty_assessments_all_data_40000_60000.parquet")
dataset4 = Path("novelty_dataset_1.0/generated_novelty_assessments_all_data_60000_80000.parquet")


df1 = pl.read_parquet(dataset1)
df2 = pl.read_parquet(dataset2)
df3 = pl.read_parquet(dataset3)
df4 = pl.read_parquet(dataset4)


df = pl.concat([df1, df2, df3, df4], how="vertical")
df_with_novelty = df.with_columns([
    pl.col("generated_text")
    .str.replace_all("*", "", literal=True)  # Remove all asterisks first
    .str.extract(r"(?i)Novelty assessment:\s*(-?\d+)", 1)  # Extract number after "Novelty assessment:"
    .cast(pl.Int8)
    .alias("novelty_score")
])
novelty_by_paper = df_with_novelty.group_by("paper_id").agg([
    pl.col("review").alias("review"),                 # will become list(String)
    pl.col("generated_text").alias("generated_text"), # will become list(String)
    pl.col("novelty_score").alias("novelty_scores")   # will become list(Int8)
])
print(f"loaded df: {novelty_by_paper.shape}")
print(novelty_by_paper.schema)


model_name = "meta-llama/Llama-3.1-8B-Instruct"

df_test = novelty_by_paper
df_test = df_test.rename({"paper_id": "id"})

# Initialize tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, legacy=False)

# Convert Polars DataFrame to Hugging Face Dataset
ds_test = hfds.Dataset.from_polars(df_test)


SYSTEM_PROMPT_TEMPLATE = """You are an expert reviewer for top-tier AI and ML conferences. 
Your task is to analyze paper reviews focusing on novelty aspects and synthesize a single, cohesive novelty assessment.
Given multiple peer reviews of a research paper, distill the collective perspective on novelty and originality into a unified evaluation written as direct statements about the work itself.
Be factual, concise, and balanced. 
Do not add your own judgment; only reflect the points expressed in the given reviews.
Write in a direct voice about the paper (e.g., "The paper introduces...", "The work extends...", "The approach combines...") rather than describing what reviewers said.
"""

USER_PROMPT_TEMPLATE = """
You will be provided with full review texts and pre-extracted novelty-related quotes and assessments from each review. 
Read and analyze carefully the following reviews for a research paper:

Full Reviews:
{reviews}

The following segments were identified as novelty discussions for each review:
{novelty_excerpts}

Assign a novelty score (-1–2) reflecting how reviewers collectively perceive the originality of the paper. 
Base the score only on their aggregated opinions.

-1 = Not novel (Reviewers consistently describe work as incremental, derivative, or replicating existing approaches with minimal innovation.)
0 = Limited novelty (Reviewers find the work somewhat standard, showing minor variations or applications of known methods without substantial conceptual or technical innovation.)
1 = Moderately novel (Reviewers acknowledge some originality but note significant overlap with prior work, or see it as a competent extension/combination of existing ideas.)
2 = Highly novel (Reviewers recognize fundamentally new ideas, approaches, problem formulations, or insights that significantly advance the field.)

Write your assessment as direct statements about the paper itself, NOT as a summary of what reviewers said.
AVOID phrases like: "Reviewers note that...", "They agree that...", "The reviewers find...", "According to reviewers..."
USE phrases like: "The paper introduces...", "The work presents...", "The approach builds on...", "The contribution lies in..."

If reviewers disagree significantly on specific aspects, present both perspectives as contrasting assessments of the work itself (e.g., "The method extends X, though the extensions remain largely incremental").

The output should be in the following format:

Novelty Score: [-1, 0, 1, 2]

Score Justification:
[2-3 sentences explaining the novelty level as direct statements about the paper's contributions and their originality]

Detailed Assessment:
[4-6 sentences written as direct statements about the paper covering:]
- The main novel contributions or new ideas introduced
- Limitations in originality or areas of incremental advancement
- Specific aspects of novelty: problem formulation, methodological innovations, experimental insights, or theoretical advances
"""


def create_messages(row):
    """
    Creates a list of message dictionaries for the chat template,
    including system and user prompts.
    """
    reviews_formatted = "\n\n".join([f"Review {i+1}: {rev}" for i, rev in enumerate(row['review'])])
    novelty_excerpts_formatted = "\n\n".join([f"Novelty Excerpt {i+1}: {exc}" for i, exc in enumerate(row['generated_text'])])
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            reviews=reviews_formatted,
            novelty_excerpts=novelty_excerpts_formatted
        )}
    ]
    return messages

# Apply the create_messages function to the dataset
ds_test = ds_test.map(lambda row: {
    'id': row['id'],
    'review': row['review'],
    'generated_text': row['generated_text'],
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
        'paper_id': row['id'],
        'review' : row['review'],
        'generated_text': row['generated_text'],
        'input_ids': tokenized_output[0].tolist(), # Convert tensor to list for dataset
        'attention_mask': [1] * len(tokenized_output[0].tolist()) # Attention mask is all ones for the prompt
    }

# Apply the preparation function to the dataset
ds_test = ds_test.map(lambda row: prepare_for_inference(row, tokenizer))

# Select only the necessary columns for inference
ds_test = ds_test.select_columns(['paper_id', 'review', 'generated_text', 'input_ids', 'attention_mask', 'messages'])

print(f"generated df: {ds_test.shape}")
print("\n--- Dataset Schema ---")
print(ds_test)
print("\n--- Example Messages Format ---")
print(ds_test[0]['messages'])

total_rows = len(ds_test)
chunk_size = total_rows // 5

# Split and save
for i in range(5):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size if i < 4 else total_rows
    
    # Use select() to get a Dataset object, not a dict
    chunk = ds_test.select(range(start_idx, end_idx))
    chunk.to_parquet(f"inference_data/chunk_{i+1}.parquet")
    print(f"Saved chunk_{i+1}.parquet with {len(chunk)} rows")

