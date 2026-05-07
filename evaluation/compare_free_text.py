# Author: Abeer Mansour

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
import polars as pl
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("roberta-large-mnli")
mdl = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(DEVICE).eval()

def nli_discriminative_score(model_outputs, references, batch_size=32):
    model_outputs = [("" if x is None else str(x)).strip() for x in model_outputs]
    references    = [("" if x is None else str(x)).strip() for x in references]

    entail, contra = [], []
    for i in range(0, len(model_outputs), batch_size):
        p = model_outputs[i:i+batch_size]  # premise: model output
        h = references[i:i+batch_size]     # hypothesis: novelty summary (reference)
        enc = tok(p, h, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(mdl(**enc).logits, dim=-1).cpu().numpy()
        contra.extend(probs[:, 0])
        entail.extend(probs[:, 2])

    entail = np.array(entail)
    contra = np.array(contra)
    score = (float((entail - contra).mean()) + 1.0) / 2.0  # scale to [0, 1]
    return score
    

ce = CrossEncoder("cross-encoder/stsb-roberta-large")


model = SentenceTransformer('all-MiniLM-L6-v2')

def judge(model_output, novelty_summary):
    # Compute embeddings
    embedding1 = model.encode(novelty_summary)
    embedding2 = model.encode(model_output)
    
    # Compute cosine similarity
    similarity = util.cos_sim(embedding1, embedding2).item()

    return similarity
'''
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
'''
files = [
    "test_novelty_generation/test_outputs_systematic_500/full_pipeline_results.parquet"
]


'''
results = []
for file in files:
    df = pl.read_parquet(file)
    scores = [judge(row["generated_text"], row["novelty_summary"]) for row in df.iter_rows(named=True)]
    #print(f"Judged scores for model {file.split('/')[-2]}: {scores}")
    mean = sum(scores)/len(scores)
    print(f"Overall judged score for model {file.split('/')[-2]}: {mean}")
    results.append(scores)
'''
'''
for file in files:
    df = pl.read_parquet(file).head(50)
    pairs = list(zip(df["generated_text"].to_list(), df["novelty_summary"].to_list()))
    scores = ce.predict(pairs, batch_size=32)
    print(f"score for model {file.split('/')[-2]}")
    print("CE mean:", float(np.mean(scores)))
''' 

for file in files:
    df = pl.read_parquet(file)
    score = nli_discriminative_score(df["generated_text"].to_list(), df["novelty_summary"].to_list(), batch_size=32)

    print(f"score for model {file.split('/')[-2]}")
    print("score:", score)
