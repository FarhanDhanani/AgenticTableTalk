CUDA_VISIBLE_DEVICES=0 python main.py --model Qwen/Qwen2-72B-Instruct \
--key_path keys/runpod_vllm_pass.txt \
--temperature 0.0 \
--max_iteration_depth 6 \
--seed 42 \
--eval_model gpt-3.5-turbo \
--eval_model_key_path keys/openai_pass.txt \
--base_url https://94zmpqe0wq9oxc-8000.proxy.runpod.net/v1 \
--dataset ait-qa \
--qa_path dataset/AIT-QA/aitqa_clean_questions.json \
--table_folder dataset/AIT-QA/aitqa_clean_tables.json \
--embedder_path jmorgan/gte-small:latest \
--embed_cache_dir dataset/AIT-QA/ \
--start 0 \
--end 80 \



