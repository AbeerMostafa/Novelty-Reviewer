# Author: Abeer Mansour

import datasets as hfds
import polars as pl
import transformers
from tqdm.auto import tqdm
from pathlib import Path
import torch 


model_name = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = Path("novelty_dataset_2.0")
OUTPUT_DIR.mkdir(exist_ok=True)


df_test = pl.read_parquet("openreviewer_dataset/dataset.zstd.parquet")

print(f"loaded df: {df_test.shape}")
print(df_test.schema)

# Initialize tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, legacy=False)

# Convert Polars DataFrame to Hugging Face Dataset
ds_test = hfds.Dataset.from_polars(df_test)

# Define system and user prompt templates
SYSTEM_PROMPT_TEMPLATE = """You are an expert scientific assistant trained to assess academic peer reviews for discussions of novelty. Your task is to:

Analyze the reviewer's free-text feedback in the review field.

Identify explicit or implicit statements about the novelty of the paper.

Classify the novelty assessment as follows:

1 : The reviewer clearly or implicitly mentions the paper is novel.

-1 : The reviewer clearly or implicitly mentions the paper lacks novelty.

0 : There is no discussion about novelty.

When novelty is discussed (assessment = 1 or -1), include one or two brief supporting quotes from the reviewer that mention or imply novelty. If novelty is not discussed (assessment = 0), leave the quotes from the reviewer field as null.

Output must contain only the following 2 fields:

1. Novelty assessment: -1, 0, or 1.

2. Quotes from the reviewer: 1–2 short exact quotes from the review supporting the assessment, or null if not applicable.

"""

USER_PROMPT_TEMPLATE = """

{review}

"""

# Function to create messages in the chat format
def create_messages(row):
    """
    Creates a list of message dictionaries for the chat template,
    including system and user prompts.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(review=row['review'])}
    ]
    return messages

# Apply the create_messages function to the dataset
ds_test = ds_test.map(lambda row: {
    'id': row['id'],  # retain original id
    'review': row['review'],
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
        'input_ids': tokenized_output[0].tolist(), # Convert tensor to list for dataset
        'attention_mask': [1] * len(tokenized_output[0].tolist()) # Attention mask is all ones for the prompt
    }

# Apply the preparation function to the dataset
ds_test = ds_test.map(lambda row: prepare_for_inference(row, tokenizer))

# Select only the necessary columns for inference
ds_test = ds_test.select_columns(['paper_id', 'review', 'input_ids', 'attention_mask', 'messages'])

# Save the processed test dataset
ds_test.to_parquet(OUTPUT_DIR / "All_data_prepared_for_inference.parquet")
print(f"Prepared dataset saved to {OUTPUT_DIR / 'All_data_prepared_for_inference.parquet'}")
print(f"test data: {ds_test.shape}")

print("\n--- Demonstrating Chat Completions (Inference) ---")

# Load the model for generation
# Ensure you have sufficient GPU memory if running on a GPU.
# Using bfloat16 for reduced memory footprint if supported.
try:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency if supported
        device_map="auto" # Automatically map model to available devices (e.g., GPU)
    )
    print(f"Model '{model_name}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Attempting to load with default settings.")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )

# Set model to evaluation mode
model.eval()

ds_test_200 = ds_test.select(range(200))

# List to store the generated results
generated_results = []

for i, example in enumerate(tqdm(ds_test_200, desc="Generating completions")):
    
    paper_id = example['paper_id']
    original_review = example['review']

    input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(model.device) # Add batch dimension and move to model device
    attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(model.device)

    print(f"\n--- Generating for Example {i+1} (Paper ID: {paper_id}) ---")
    # print("\nInput Prompt:")
    # print(tokenizer.decode(input_ids[0], skip_special_tokens=False))

    # Generate completion
    # max_new_tokens: Limits the length of the generated response
    # do_sample: If True, uses sampling for generation; if False, uses greedy decoding
    # top_p, temperature: Parameters for controlling sampling diversity
    # eos_token_id: Stop generation when end-of-sentence token is encountered
    with torch.no_grad(): # Disable gradient calculation for inference
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100, # Max tokens to generate for the response
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id # Stop generation at EOS token
        )

    # Decode the generated output
    # The output_ids will contain the input_ids + the generated tokens.
    # We slice to get only the newly generated tokens.
    generated_text = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)

    # print("\nGenerated Completion:")
    # print(generated_text)
    # Print the full conversation including the generated part
    # full_conversation = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    # print("\nFull Conversation:")
    # print(full_conversation)

    # Store the results
    generated_results.append({
        "paper_id": paper_id,
        "review": original_review,
        "generated_text": generated_text
    })

print("\n--- Chat Completion Completed on Test Dat ---")

# Save the generated output to a Parquet file

df_output = pl.DataFrame(generated_results)
output_parquet_path = OUTPUT_DIR / "generated_novelty_assessments_all_data_200.parquet"
df_output.write_parquet(output_parquet_path)
print(f"\nGenerated output saved to {output_parquet_path}")


