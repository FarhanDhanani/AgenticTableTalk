# AGENTIC TABLE TALK

[![N|Solid](https://upload.wikimedia.org/wikipedia/en/e/e4/National_University_of_Computer_and_Emerging_Sciences_logo.png)](https://nodesource.com/products/nsolid)


# 📄 VSCode Launch Configuration: Argument Documentation

This document explains the arguments used in the `launch.json` configuration for launching the table QA pipeline script.

## 🧠 LLM Model Configuration

| Argument | Description |
|---------|-------------|
| `--model` | Specifies the primary LLM used for QA. Options include:<br>• `gpt-3.5-turbo`<br>• `gpt-4o`<br>• `llama3`<br>• `Qwen/Qwen2-72B-Instruct`<br>• `google/gemma-3-12b-it`<br>• `gemini-2.5-flash` *(currently active)* |
| `--key_path` | Path to the API key file used for the selected model.<br>e.g., `keys/gemni_pass.txt` for Gemini |
| `--base_url` | (Optional) Custom base URL for hosted models (e.g., VLLM, Ollama, OpenRouter). Uncomment when using remote inference APIs. |

## 🧲 Embedder Configuration

| Argument | Description |
|---------|-------------|
| `--embedder_path` | Path or name of the embedding model. <br>e.g., `jmorgan/gte-small:latest` *(currently active)* |
| `--embedder_model_key_path` | API key path for embedder (if required).<br>e.g., `keys/openai_pass.txt` |
| `--embedder_base_path` | (Optional) Path to locally stored embedder model (if applicable). |

## ⚙️ General Settings

| Argument | Description |
|----------|-------------|
| `--temperature` | Sampling temperature for the LLM. `0.0` for deterministic output. |
| `--max_iteration_depth` | Controls recursive depth for multi-step QA. |
| `--seed` | Random seed for reproducibility. |
| `--eval_model` | Model used to evaluate LLM outputs (e.g., `gpt-3.5-turbo`). |
| `--eval_model_key_path` | API key for the evaluation model. |
| `--start` / `--end` | Useful for debugging or partial runs (e.g., `--end 1` to stop after 1st item). |

## 📊 Dataset Configuration

### AIT-QA Dataset (currently active)

| Argument | Description |
|----------|-------------|
| `--dataset` | Dataset name: `ait-qa` |
| `--qa_path` | Path to the question file: `dataset/AIT-QA/aitqa_clean_questions.json` |
| `--table_folder` | Path to tables file: `dataset/AIT-QA/aitqa_clean_tables.json` |
| `--embed_cache_dir` | Directory for caching embedding results: `dataset/AIT-QA/` |

### Other Supported Datasets

Uncomment and adjust the following based on the dataset you want to use:

#### HITAB
```bash
--dataset hitab
--qa_path dataset/hitab/test_samples.jsonl
--table_folder dataset/hitab/raw/
--embed_cache_dir dataset/hitab/
```

#### Synthetic (AIT-QA)
```bash
--dataset ait-qa
--qa_path dataset/synthetic/up_generated_large_table_questions.json
--table_folder dataset/synthetic/up_large_tables/
--embed_cache_dir dataset/synthetic/
```

## 🛠️ Preprocessing & Output Options

| Argument | Description |
|----------|-------------|
| `--pre_process_tables` | Preprocess all input tables before inference. |
| `--save_markdown` | Save converted markdown tables. |
| `--generate_meta` | Generate meta-information from tables. |
| `--exit_after_preprocess` | Exit after table preprocessing is complete. Useful for preparation-only mode. |

---

## ✅ Example Launch

Here's an example `args` subset for using Gemini with AIT-QA:
```json
"args": [
  "--model", "gemini-2.5-flash",
  "--key_path", "keys/gemni_pass.txt",
  "--embedder_path", "jmorgan/gte-small:latest",
  "--embedder_model_key_path", "keys/openai_pass.txt",
  "--temperature", "0.0",
  "--max_iteration_depth", "6",
  "--seed", "42",
  "--eval_model", "gpt-3.5-turbo",
  "--eval_model_key_path", "keys/openai_pass.txt",
  "--end", "1",
  "--dataset", "ait-qa",
  "--qa_path", "dataset/AIT-QA/aitqa_clean_questions.json",
  "--table_folder", "dataset/AIT-QA/aitqa_clean_tables.json",
  "--embed_cache_dir", "dataset/AIT-QA/"
]
```


