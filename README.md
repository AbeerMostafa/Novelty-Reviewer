# Novelty-Reviewer

## Overview

This repository implements tools and analysis code for evaluating originality in AI research. It is aligned with the paper:

"Are We Truly Innovating? A Qualitative and Quantitative Analysis of AI Research Originality"

The work explores how originality is judged in peer review, builds structured prior-work comparison pipelines, and evaluates whether current LLM agents can reliably assess novelty or detect conceptual plagiarism.

## What this repo does

- Builds a discovery and analysis pipeline for research novelty and originality.
- Extracts structured knowledge from papers and constructs similarity graphs.
- Generates and evaluates novelty assessments for research submissions.
- Compares model-generated novelty scores with human/ground-truth labels.
- Provides reusable scripts for data collection, conversion, knowledge extraction, and evaluation.

## Repository structure

- `scripts/`
  - `arxiv_search.py`: Fetch recent AI papers from arXiv and download PDFs.
  - `convert_pdf_to_md.py`: Convert local PDFs into Markdown for downstream processing.
  - `llm_knowledge_extract.py`: Use an LLM to extract structured paper features and build knowledge graphs.
  - `graph_analysis.py`: Compute paper similarity and surface concept overlap using extracted features.
  - `Dataset_construction/D_01_prepare_inference_data.py`: Prepare inference datasets from aggregated novelty assessments.
  - `Novelty_Reviewer_full_pipline.py`: Run the full manuscript analysis pipeline, including knowledge extraction, semantic search, and novelty evaluation.

- `evaluation/`
  - `detect_novelty.py`: Run novelty scoring on a dataset and generate evaluation outputs.
  - `compare_novelty_scores.py`: Compare predicted novelty scores against ground truth and compute metrics.
  - `llm_as_judge.py`: Score model outputs with an LLM-based judge using semantic similarity and NLI.
  - `generate_novelty_test_data.py`: Generate test inputs for novelty evaluation.
  - `run_chat_inference_one_example.py`: Example chat-based inference for a single paper.
  - `run_test_HF.py`: Example Hugging Face inference test harness.
  - `run_test_OpenRouter.py`: Example OpenRouter inference test harness.
  - `run_hpc_test.sbatch`: HPC job submission template for batch evaluation.

- `llm_training/`
  - Fine-tuning and training config files for model training and evaluation.

- `deepspeed_configs/`
  - DeepSpeed configuration files for large-scale model training.

## Quick start

1. Clone the repository:

```bash
git clone <repo-url> Novelty-Reviewer
cd Novelty-Reviewer
```

2. Install Python dependencies. Example:

```bash
python -m pip install -r requirements.txt
```

If no requirements file is present, install the most common packages used by the repository:

```bash
python -m pip install datasets polars transformers sentence-transformers torch networkx scikit-learn matplotlib tqdm requests arxiv PyPDF2 pymupdf4llm
```

3. Run a simple test data collection step:

```bash
python scripts/arxiv_search.py
```

4. Convert downloaded PDFs to Markdown:

```bash
python scripts/convert_pdf_to_md.py
```

5. Extract structured paper knowledge and build a graph:

```bash
python scripts/llm_knowledge_extract.py
python scripts/graph_analysis.py
```

6. Run the full novelty review pipeline:

```bash
python scripts/Novelty_Reviewer_full_pipline.py
```

7. Evaluate novelty scoring and compare models:

```bash
python evaluation/compare_novelty_scores.py
python evaluation/llm_as_judge.py
```

## Notes on usage

- `Novelty_Reviewer_full_pipline.py` expects a Parquet dataset file such as `test_dataset.parquet` in the current working directory.
- `detect_novelty.py` expects the dataset at `openreviewer_dataset/dataset.zstd.parquet` and writes output into `novelty_dataset_2.0`.
- `compare_novelty_scores.py` compares model outputs in `test_novelty_generation/test_outputs_systematic_500/` and requires Parquet result files.
- `llm_knowledge_extract.py` relies on an OpenRouter API key or compatible LLM endpoint to extract JSON features from paper text.



## Note

- Use GPU-enabled PyTorch where available for efficient LLM inference.
