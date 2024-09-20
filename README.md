![Project Status](https://img.shields.io/badge/Project-Completed-brightgreen)

<div style="padding:15px 20px 20px 20px;border-left:3px solid blue;background-color:#e4f0fa;border-radius: 20px;color:#424242;">

## **Natural Language Generation Techniques**

This project explores advanced Natural Language Generation (NLG) methodologies, focusing on text generation using multiple approaches. It covers decoding algorithms, the impact of their parameters, and evaluates NLG with machine translation metrics.

### Project Overview
- **Part 1:** Implemented two decoding algorithms (greedy and beam search) alongside two sampling methods (top-p and top-k) to replicate aspects of Huggingface's `generate` function.
- **Part 2:** Developed Contrastive Decoding, which combines logits from two models during generation for improved text quality.
- **Part 3:** Evaluated NLG performance in machine translation using various metrics.

### Table of Contents
- **Setup Instructions**
  
- **Part 1: Decoding and Sampling Algorithms**
  - 1.1 Implement and test decoding and sampling algorithms
  
- **Part 2: Contrastive Decoding**
  - 2.1 Implement contrastive decoding with adaptive plausibility constraints
  - 2.2 Evaluate generations using the MAUVE metric
  
- **Part 3: Machine Translation (MT) Evaluation**
  - 3.1 Dataset and metrics analysis
  - 3.2 NLG metric calculation and correlation analysis

### Codebase Structure

- Main Jupyter notebook: `a3_notebook.ipynb`
  
- Python files (if implemented):
  - `a3_utils.py`: Helper functions
  - `a3_decoding.py`
  - `a3_sampling.py`
  - `a3_contrastive_decoding.py`
  - `a3_contrastive_main.py`
  - `a3_mt_eval.py`
  
- Q&A Document: `a3_mt_qa.md`

- Generated JSON files:
  - `part2_contrastive_generations.json`
  - `part2_greedy_generations.json`
  - `part3_metrics.json`
  - `part3_corr.json`
