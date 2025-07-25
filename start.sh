#!/bin/bash

# Enable error reporting
set -e

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=debug

# Main command
python main.py \
  --embedder_path Alibaba-NLP/gte-base-en-v1.5 \
  --temperature 0.0 \
  --max_iteration_depth 6 \
  --seed 42 \
  --eval_model gpt-3.5-turbo \
  --eval_model_key_path keys/openai_pass.txt \
  --end 1 \
  --model google/gemma-3-4b-it \
  --key_path keys/openrouter_pass.txt \
  --base_url https://openrouter.ai/api/v1 \
  --dataset ait-qa \
  --qa_path dataset/synthetic/up_generated_large_table_questions.json \
  --table_folder dataset/synthestic/generated_large_tables/ \
  --embed_cache_dir dataset/synthetic/
