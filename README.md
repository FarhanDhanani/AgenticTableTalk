[![N|Solid](https://upload.wikimedia.org/wikipedia/en/e/e4/National_University_of_Computer_and_Emerging_Sciences_logo.png)](https://nodesource.com/products/nsolid)

# TITLE: AGENTIC TABLE TALK

# Description
This repository presents a novel approach for solving closed-domain, free-form, non-semantic tabular question answering using an Agentic ReAct paradigm. It leverages general-purpose LLMs (e.g., GPT, Gemini, Qwen) in combination with a greedy, heuristic based on Retrieval-Augmented Generation (RAG) architecture which priotizes local neighbourhood of a particular table cell along with its relevancy to the query.

The inspiration of the proposed implementation was taken from Decoupling Data and Linguistic Knowledge for Spontaneous Tabular Q/A, GraphOTTER: Evolving LLM-based Graph Reasoning for Complex Table Question Answering,  Table Talk Papers cited below. The main addition that this repository showcases is the combination of Agentic ReAct paradigm with greedy RAG base heuristic which priotizes local neighbourhood of a table cell along with its relvancy to the given query. Instead of processing entire table each time for a given query, the proposed implementation iteratively explores the regions of the table by focusing on the table cells and their neighbours that more more relevant to the given query until the required facts to answer the query are known. This simple idea keeps the length of input context manageable and saves from input token limit-related exception specially in the case of large tables when passing entire table to a LLM for each incoming query is not sensible.

```bibtex
@INPROCEEDINGS{IBCAST,
  author={Dhanani, Farhan and Rafi, Muhammad},
  booktitle={2023 20th International Bhurban Conference on Applied Sciences and Technology (IBCAST)}, 
  title={Decoupling Data and Linguistic Knowledge for Spontaneous Tabular Q/A}, 
  year={2023},
  volume={},
  number={},
  pages={226-231},
  keywords={Deep learning;Computer science;Scalability;Production;Linguistics;Real-time systems;Natural language processing;Hardware;Data mining;Optimization;Open AI;LLMs;Table Question-Answering (Q/A);Pinecone;Vector Database;RETRO},
  doi={10.1109/IBCAST59916.2023.10713035}},

@misc{li2024graphotter,
      title={GraphOTTER: Evolving LLM-based Graph Reasoning for Complex Table Question Answering}, 
      author={Qianlong Li and Chen Huang and Shuai Li and Yuanxin Xiang and Deng Xiong and Wenqiang Lei},
      year={2024},
      eprint={2412.01230},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.01230}, 
}

@INPROCEEDINGS{INMIC-TableTalk,
  author={Dhanani, Farhan and Rafi, Muhammad},
  booktitle={2024 26th International Multi-Topic Conference (INMIC)}, 
  title={Table Talk}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  keywords={Social networking (online);Scalability;Retrieval augmented generation;Production;Computer architecture;Transformers;Vectors;Real-time systems;Question answering (information retrieval);Hardware;LLMs;Tabular Question Answering;Pinecone;Vector Database},
  doi={10.1109/INMIC64792.2024.11004342}}
```


# Dataset Information
The resposiotry uses benchmark test sets of Hitab and AIT-QA datasets to evulate the performance of the proposed approach along with synthetically generated datasets containing large tables.


## 📚 Citation for Hitab

If you use this repository in your work, please cite the following datasets:

```bibtex
@article{cheng2021hitab,
  title={HiTab: A Hierarchical Table Dataset for Question Answering and Natural Language Generation},
  author={Cheng, Zhoujun and Dong, Haoyu and Wang, Zhiruo and Jia, Ran and Guo, Jiaqi and Gao, Yan and Han, Shi and Lou, Jian-Guang and Zhang, Dongmei},
  journal={arXiv preprint arXiv:2108.06712},
  year={2021}
}
```
## 📚 Citation for AIT-QA
```bibtex
@misc{katsis2021aitqa,
  title={AIT-QA: Question Answering Dataset over Complex Tables in the Airline Industry}, 
  author={Yannis Katsis and Saneem Chemmengath and Vishwajeet Kumar and Samarth Bharadwaj and Mustafa Canim and Michael Glass and Alfio Gliozzo and Feifei Pan and Jaydeep Sen and Karthik Sankaranarayanan and Soumen Chakrabarti},
  year={2021},
  eprint={2106.12944},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

# Code Information
- The code implementation is solely based on python 3.11. The code can be deployed on any system where python 3.11 runtime is available. To run the code implementation there is no speciall need of any expensive hardware or gpu. The code internally uses open-ai client's implementation to make an API call to the specified **LLM** at the mentioned **base-url**. Therefore all the Unified LLM Platforms which supports the open-ai specification, such as OpenRouter, Requesty, ollama, litellm, etc are supported.

- Additionally, the current implementation also supports custom LLM models. They can be deployed via VLLM either on gpu renting platforms like (runpod, aws, google cloud) or at local to integrate with proposed implementation.

- Lastly, the current implementatioin also supports gemni, and provide a sperate client interface to connect with google models.

- The detail implementation and configurations of the discussed clients can be found under the **Generator/** directory.

- The current implementation supports multiple methods for loading embedders. Below is the list of supported options:
 
    - Load embedder via OpenAI Client (Recommended): This option allows integration of open-ai embedders and the embedders avalable at any Frameworks or Platforms which supports the open-ai specification.

    - Load embedder via Ollama: This is usefull for the embedder available at ollama platform.

    - Load embedder via sentence-transformer or AutoTokenizer: This is use-full for custom fine-tuned embedders based on sentence-transfomrer's/autoTokenizer's implementation. However, please note this option will also require appropriate hardware support as the embedder will loaded from local enviroment.

        - The correct place to place custom embedders is in ./model directory and specify the path appropriately in command line argument called **embedder_path**.

- The current implementation can be extended to support for more methods to load embedder by extending or modifying exisiting **init** method of **Retriever** class present in *./GraphRetriever/dense_retriever.py* file

---

# Usage Instruction

- ## 🐍 Python Version Setup 

    ⚠️ This repository requires Python version 3.11.8 to run correctly.

    Here's a concise README snippet for setting up **Python 3.11.8 via `pyenv`**:

    PYTHON VERSION: 3.11.8
    
    This project requires **Python 3.11.8**.

    #### 🔧 Setup via `pyenv`

    1. **Install `pyenv`** (if not already installed):
    👉 [macOS installation guide](https://mac.install.guide/python/install-pyenv)

    2. **Install Python 3.11.8**:

    ```bash
    pyenv install 3.11.8
    ```

    3. **Set local Python version**:

    ```bash
    pyenv local 3.11.8
    ```

    4. **Verify**:

    ```bash
    python --version 
    # OR 
    python3 --version
    # Should output: Python 3.11.8
    ```
- ## 🧪 Setting Up Virtual Environment

    Create and activate a virtual environment named `gOTTER_env`:

    ```bash
    # Step 1: Create the virtual environment
    python3 -m venv gOTTER_env

    # Step 2: Activate the virtual environment
    source gOTTER_env/bin/activate
    ```

    > 📝 You should see the environment name `(gOTTER_env)` appear in your terminal prompt after activation.
    > To deactivate later, simply run:
    >
    > ```bash
    > deactivate
    > ```

- ## 📂 Extract Zipped Data

    To extract the zipped data located at `dataset/hitab/raw/`, run the following command:

    ```bash
    python zip_extractor.py
    # OR
    python3 zip_extractor.py
    ```

    > 📝 Make sure the `zip_extractor.py` script is placed in the same directory as the zip file (`dataset/hitab/raw/`), and the zip file (e.g. `raw-tables.zip`) is present in that folder.

- ## 🔐 Setting Up API Keys

    To authenticate with various services used in this project, create the following files in the `keys/` directory and place the appropriate API key/token inside each file.

    ### 📁 Navigate to the `keys/` directory:

    ```bash
    cd keys/
    ```

    ### ✏️ Create and populate the required key files:

    ```bash
    touch gemni_pass.txt           # Required if using Gemini models from Google
    touch huggingface_pass.txt     # Required if using Hugging Face embeddings or running `count_db_tokens.py`
    touch ollama_key.txt           # Required if using LLMs or embedders hosted via Ollama
    touch openai_pass.txt          # Required in all cases (used for GPT-3.5-Turbo)
    touch openrouter_pass.txt      # Required if using LLMs hosted via OpenRouter
    touch runpod_vllm_pass.txt     # Required if using LLMs hosted via RunPod or other GPU services with vLLM
    ```

    > 💡 **Add your respective API key/token in plain text inside each `.txt` file.**

    For example:

    ```bash
    echo "your_openai_key_here" > openai_pass.txt
    ```
- ## 🧠 Loading Local LLM or Embedder Models (OPTIONAL)

    If you want to use a **local LLM model or embedder** and load via libraries like sentence-transformers, place the model files inside the `model/` directory at the root of the project.

    ```
    project-root/
    ├── model/
    │   └── your-local-model-files-here
    ```

    > 📦 The application will automatically check the `model/` directory when loading local models.
    > Make sure the model format is compatible with the framework you're using (e.g., sentence-transformers, HuggingFace, etc.)

---

# 📦 Requirements: Installing Dependencies

    
```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt
```


Make sure the following Python dependencies are installed. It's recommended to use a **virtual environment** with Python **3.11.8**

### ✅ Basic Installation (CPU-only)

```bash
pip install jsonlines
pip install xlrd
pip install pandas
pip install jieba
pip install nltk
pip install openai
pip install faiss-cpu        # For CPU systems
pip install torch            # For CPU systems
pip install sentence-transformers
pip install langchain-ollama
pip install google-generativeai
pip install protobuf
pip install google-cloud-aiplatform
pip install sentencepiece
```

### ⚡️ Optional: GPU Support

If you intend to use **FAISS with GPU** and **PyTorch with GPU**, use:

```bash
pip install faiss-gpu  # Only if you want GPU-based FAISS
```

**Install GPU-compatible Torch** by replacing the placeholder below with the appropriate URL for your GPU and system:
➡️ [Official PyTorch install guide](https://pytorch.org/get-started/locally/)

```bash
pip install torch torchvision torchaudio --index-url <URL_FROM_PYTORCH_GUIDE>
```
---

# Pre-Processing Methodologies
The **pre_process_tables** function present in the main file, does the actual pre-processing to generate a markdown for each of the table present in the dataset to generate their summaries. The pre-calculation of summary files is important to boot-up the proposed approach.

The pre-processing section is controled via following commandline arguments 

```bash
"--model", "gpt-4o", %name of model which generates the summaries of the tables
"--pre_process_tables","true", %flag to start pre-processing the tables.
"--save_markdown", "true", %flag to save the markdown file generated from after converting the table
"--generate_meta", "true", %flag to generate the meta information extracted from the markdown table
"--exit_after_preprocess" %flag to exit the application after extracting meta information
```

# Guide To Run the Application 
## 📄 VSCode Launch Configuration: Run Project in VS-CODE Via launch.json

This section explains the arguments used in the `launch.json` configuration for launching the table QA pipeline script.

### 🧠 LLM Model Configuration

| Argument | Description |
|---------|-------------|
| `--model` | Specifies the primary LLM used for QA. Options include:<br>• `gpt-3.5-turbo`<br>• `gpt-4o`<br>• `llama3`<br>• `Qwen/Qwen2-72B-Instruct`<br>• `google/gemma-3-12b-it`<br>• `gemini-2.5-flash` *(currently active)* |
| `--key_path` | Path to the API key file used for the selected model.<br>e.g., `keys/gemni_pass.txt` for Gemini |
| `--base_url` | (Optional) Custom base URL for hosted models (e.g., VLLM, Ollama, OpenRouter). Uncomment when using remote inference APIs. |

### 🧲 Embedder Configuration

| Argument | Description |
|---------|-------------|
| `--embedder_path` | Path or name of the embedding model. <br>e.g., `jmorgan/gte-small:latest` *(currently active)* |
| `--embedder_model_key_path` | API key path for embedder (if required).<br>e.g., `keys/openai_pass.txt` |
| `--embedder_base_path` | (Optional) Path to locally stored embedder model (if applicable). |

### ⚙️ General Settings

| Argument | Description |
|----------|-------------|
| `--temperature` | Sampling temperature for the LLM. `0.0` for deterministic output. |
| `--max_iteration_depth` | Controls recursive depth for multi-step QA. |
| `--seed` | Random seed for reproducibility. |
| `--eval_model` | Model used to evaluate LLM outputs (e.g., `gpt-3.5-turbo`). |
| `--eval_model_key_path` | API key for the evaluation model. |
| `--start` / `--end` | Useful for debugging or partial runs (e.g., `--end 1` to stop after 1st item). |

### 📊 Dataset Configuration

#### AIT-QA Dataset (currently active)

| Argument | Description |
|----------|-------------|
| `--dataset` | Dataset name: `ait-qa` |
| `--qa_path` | Path to the question file: `dataset/AIT-QA/aitqa_clean_questions.json` |
| `--table_folder` | Path to tables file: `dataset/AIT-QA/aitqa_clean_tables.json` |
| `--embed_cache_dir` | Directory for caching embedding results: `dataset/AIT-QA/` |

#### Other Supported Datasets

Uncomment and adjust the following based on the dataset you want to use:

##### HITAB
```bash
--dataset hitab
--qa_path dataset/hitab/test_samples.jsonl
--table_folder dataset/hitab/raw/
--embed_cache_dir dataset/hitab/
```

##### Synthetic (AIT-QA)
```bash
--dataset ait-qa
--qa_path dataset/synthetic/up_generated_large_table_questions.json
--table_folder dataset/synthetic/up_large_tables/
--embed_cache_dir dataset/synthetic/
```

### 🛠️ Preprocessing & Output Options

| Argument | Description |
|----------|-------------|
| `--pre_process_tables` | Preprocess all input tables before inference. |
| `--save_markdown` | Save converted markdown tables. |
| `--generate_meta` | Generate meta-information from tables. |
| `--exit_after_preprocess` | Exit after table preprocessing is complete. Useful for preparation-only mode. |

---

### ✅ Example Launch

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

---

## 🚀 Running the Application via Command Line (for MAC/Linux users)

You can run the entire application from the command line by properly configuring the `start.sh` script with the required arguments.

### 🛠️ Steps:

1. **Open `start.sh`** and update the placeholder variables or command-line flags as needed for your setup (e.g., model paths, data files, modes, etc.).

2. **Make the script executable** (if not already):

    ```bash
    chmod +x start.sh
    ```

3. **Run the script:**

    ```bash
    ./start.sh
    ```

    > ✅ Ensure your virtual environment is activated and all dependencies are installed before running the script.

    ---