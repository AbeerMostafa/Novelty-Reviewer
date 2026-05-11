# Novelty-Reviewer

## Overview

This repository contains the benchmark dataset and source code for the paper:

"Are We Truly Innovating? A Qualitative and Quantitative Analysis of AI Research Originality"

The work explores how originality is judged in peer review, builds structured prior-work comparison pipelines, and evaluates whether current LLM agents can reliably assess novelty or detect conceptual plagiarism.

## Repository structure

- `scripts/`
  - `convert_pdf_to_md.py`: Convert local PDFs into Markdown for downstream processing.
  - `llm_knowledge_extract.py`: Use an LLM to extract structured paper features and build knowledge graphs.
  - `graph_analysis.py`: Compute paper similarity and surface concept overlap using extracted features.
  - `Dataset_construction/D_01_prepare_inference_data.py`: Prepare inference datasets from aggregated novelty assessments.
  - `Novelty_Reviewer_full_pipline.py`: Run the full manuscript analysis pipeline, including knowledge extraction, semantic search, and novelty evaluation.

- `evaluation/`
  - `compare_novelty_scores.py`: Compare predicted novelty scores against ground truth and compute metrics.
  - `llm_as_judge.py`: Score model outputs with an LLM-based judge using semantic similarity and NLI.
  - `generate_novelty_test_data.py`: Generate test inputs for novelty evaluation.
  - `run_chat_inference_one_example.py`: Example chat-based inference for a single paper.
  - `run_test_HF.py`: Example Hugging Face inference test.
  - `run_test_OpenRouter.py`: Example OpenRouter inference test.
  - `run_hpc_test.sbatch`: HPC job submission template for batch evaluation.

- `llm_training/`
  - Fine-tuning and training config files for model training and evaluation.

- `deepspeed_configs/`
  - DeepSpeed configuration files for large-scale model training.

## Quick start

1. Install Python dependencies. Example:

```bash
python -m pip install -r requirements.txt
```

2. Run a simple test data collection step:

```bash
python scripts/arxiv_search.py
```

3. Convert downloaded PDFs to Markdown:

```bash
python scripts/convert_pdf_to_md.py
```

4. Run the full novelty review pipeline:

```bash
python scripts/Novelty_Reviewer_full_pipline.py
```

5. Evaluate novelty scoring and compare models:

```bash
python evaluation/compare_novelty_scores.py
python evaluation/llm_as_judge.py
```

## Notes on usage

- `compare_novelty_scores.py` compares model outputs in `test_novelty_generation/test_outputs_systematic_500/` and requires Parquet result files.
- You should provide your own API keys where necesary.
- Use GPU-enabled PyTorch where available for efficient LLM inference.
