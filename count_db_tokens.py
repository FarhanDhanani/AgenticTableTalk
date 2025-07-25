import os
import json
import jsonlines
import warnings
import tiktoken
from enum import Enum
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vertexai.preview import tokenization
from huggingface_hub import login, whoami
from transformers import Qwen2TokenizerFast
from GraphRetriever.graph_retriver import GraphRetriever
from huggingface_hub.utils import RepositoryNotFoundError
from tools import table2Tuple, read_pre_process_tabular_meta
from GraphRetriever.dense_retriever import load_dense_retriever
from GraphRetriever.table_reader import hitab_table_converter,ait_qa_converter

hugging_face_token_path = "./keys/huggingface_pass.txt"
USE_OLLAMA_EMBEDDER = "jmorgan/gte-small:latest"

class DataSet(Enum):
    HITAB = 'hitab'
    AIT_QA = 'ait-qa'
    SYNTHETIC = 'synthetic'

class Models(Enum):
    GEMINI_1_5_FLASH =  "gemini-1.5-flash"
    QWEN2 = 'Qwen/Qwen2-7B-Instruct'
    QWEN3_14b = 'Qwen/Qwen3-14B'
    GEMMA3_12b = 'google/gemma-3-12b-it'
    QWEN3_8b = 'Qwen/Qwen3-8B'
    GEMMA3_4b = 'google/gemma-3-4b-it'
    CHAT_GPT = 'gpt-4o'

class Args:
    def __init__(self, dataset, qa_path, table_folder, embed_cache_dir,model, start, end, embedder_path):
        self.model = model
        self.embedder_path = embedder_path  
        self.dataset = dataset
        self.qa_path = qa_path
        self.table_folder = table_folder
        self.start = start
        self.end = end
        self.embed_cache_dir = embed_cache_dir
        self.gpu = False

def login_hugging_face_with_token(file_path):
    try:
        # Read the token from the file
        with open(hugging_face_token_path, "r") as file:
            hf_token = file.read().strip()

        if not hf_token:
            raise ValueError("Error: Token file is empty!")

        # Attempt to log in
        login(hf_token)

        # Verify if login was successful
        user_info = whoami()  # Will raise an error if login fails
        print(f"✅ Successfully logged in as: {user_info['name']}")

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{file_path}' not found!")
    except RepositoryNotFoundError:
        raise ValueError("Error: Invalid Hugging Face token! Please check your token.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")   
    return

def get_tokenizer(model):
    if model == Models.GEMINI_1_5_FLASH:
        tokenizer = tokenization.get_tokenizer_for_model(model.value)

    elif model in {
        Models.QWEN2, Models.QWEN3_14b, Models.QWEN3_8b,
        Models.GEMMA3_12b, Models.GEMMA3_4b
    }:
        login_hugging_face_with_token(hugging_face_token_path)
        tokenizer = AutoTokenizer.from_pretrained(model.value)

    elif model == Models.CHAT_GPT:
        tokenizer = tiktoken.encoding_for_model(model.value)

    else:
        raise ValueError("Unsupported model type!")
    return tokenizer

def count_tokens(text, tokenizer):
    """Counts the number of tokens in the given text using the provided tokenizer."""
    if isinstance(tokenizer, Qwen2TokenizerFast):  # For Qwen models
        tokens = tokenizer(text, truncation=False, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        return len(tokens)
    elif isinstance(tokenizer, PreTrainedTokenizerBase):  # For Hugging Face models
        return len(tokenizer(text)["input_ids"])
    elif hasattr(tokenizer, "count_tokens"):  # Specially For Gemini
        return tokenizer.count_tokens(text).total_tokens
    elif isinstance(tokenizer, tiktoken.Encoding):  # OpenAI's tiktoken
        return len(tokenizer.encode(text))
    else:
        raise ValueError("Unsupported tokenizer type!")

def load_data(args):

    querys,answers,table_captions,tables,table_paths = [],[],[],[],[]

    if args.dataset.lower() in (DataSet.HITAB.value):
        qas = []
        with open(args.qa_path, "r+", encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                qas.append(item)
        qas = qas[args.start:args.end]

        for qa in qas:
            table_path = args.table_folder + qa['table_id'] + '.json'
            with open(table_path, "r+", encoding='utf-8') as f:
                table = json.load(f)
            table_captions.append(table['title'])
            answers.append('|'.join([str(i) for i in qa['answer']]))
            querys.append( qa['question'])
            table_paths.append(table_path)
            tables.append(table['texts'])
            
    elif args.dataset.lower() in (DataSet.AIT_QA.value):
        with open(args.qa_path, 'r', encoding='utf-8') as f:
            qas = json.load(f)
        qas = qas[args.start:args.end]

        for qa in qas:
            tables.append(qa['table'])
            answers.append('|'.join([str(i) for i in qa['answers']]))
            querys.append( qa['question'])
            table_captions.append('')
            table_paths.append(qa)
        
    elif args.dataset.lower() in (DataSet.SYNTHETIC.value):
        with open(args.qa_path, 'r', encoding='utf-8') as f:
            qas = json.load(f)
        qas = qas[args.start:args.end]

        for qa in qas:
            tables.append(qa['table'])
            answers.append('|'.join([str(i) for i in qa['answers']]))
            querys.append(qa['question'])
            table_captions.append('')  # Synthetic tables have no title
            table_paths.append(qa)  
    else:
        raise ValueError('The dataset is not supported')
    
    return querys,answers,table_captions,tables,table_paths

def build_args_for_ait_qa(model:Models, start=0, end=None):
    ait_qa_args = Args(dataset=DataSet.AIT_QA.value, 
                       qa_path="dataset/AIT-QA/aitqa_clean_questions.json", 
                       table_folder="dataset/AIT-QA/aitqa_clean_tables.json",
                       embed_cache_dir="dataset/AIT-QA/",
                       model=model.value if model else None,
                       start=start, 
                       end=end,
                       embedder_path=USE_OLLAMA_EMBEDDER)

    return ait_qa_args

def build_args_for_hitab(model:Models, start=0, end=None):
    hitab_args = Args(dataset=DataSet.HITAB.value, 
                      qa_path="dataset/hitab/test_samples.jsonl", 
                      table_folder="dataset/hitab/raw/",
                      embed_cache_dir="dataset/hitab/",
                      model=model.value if model else None,
                      start=start,
                      end=end,
                      embedder_path=USE_OLLAMA_EMBEDDER)
   
    return hitab_args

def build_args_for_synthetic(model:Models, start=0, end=None):
    synthetic_args = Args(dataset=DataSet.SYNTHETIC.value, 
                          qa_path="dataset/synthetic/up_generated_large_table_questions.json", 
                          table_folder="dataset/synthetic/generated_large_tables/",
                          embed_cache_dir="dataset/synthetic/",
                          model=model.value if model else None,
                          start=start, 
                          end=end,
                          embedder_path=USE_OLLAMA_EMBEDDER)
    
    return synthetic_args


def compute_avg_table_tokens(querys, answers, table_captions, tables, table_paths, tokenizer, table_converter):
    total_tokens = 0
    total_records =0
    for query, answer, caption, table, table_path in zip(querys, answers, table_captions, tables, table_paths):
        dealed_rows, dealed_cols, rows, cols, merged_cells = table_converter(table_path)
        #cells = table2Tuple(dealed_rows)
        if caption:
            table = f"Table Caption: {caption} \n**Table:**\n {str(dealed_rows)}"
        else:
            table = f"**Table:**\n{str(dealed_rows)}"
        
       
        tokens = count_tokens(table, tokenizer=tokenizer)
        total_tokens += tokens
        total_records += 1 
    
    return total_tokens/total_records

def compute_avg_rows_cols_for_each_dataset(tables):
    total_tables = 0
    rows_count = 0
    cols_count = 0

    for table in tables:
        rows_count += len(table)
        cols_count += len(table[0])
        total_tables += 1

    return (rows_count / total_tables), (cols_count / total_tables)

def compute_avg_table_tokens_for_each_model(model: Models, start=0, end=None):
    tokenizer = get_tokenizer(model)
    
    
    querys, answers, table_captions, tables, table_paths = load_data(build_args_for_ait_qa(model, start, end))
    tokens_ait_qa = compute_avg_table_tokens(querys, answers, table_captions, tables, table_paths, tokenizer, ait_qa_converter)
    ait_avg_row, ait_avg_col = compute_avg_rows_cols_for_each_dataset(tables)

    querys, answers, table_captions, tables, table_paths = load_data(build_args_for_hitab(model, start, end))
    tokens_hitab = compute_avg_table_tokens(querys, answers, table_captions, tables, table_paths, tokenizer, hitab_table_converter)
    hitab_avg_row, hitab_avg_col = compute_avg_rows_cols_for_each_dataset(tables)

    querys, answers, table_captions, tables, table_paths = load_data(build_args_for_synthetic(model, start, end))
    tokens_synthetic = compute_avg_table_tokens(querys, answers, table_captions, tables, table_paths, tokenizer, ait_qa_converter)
    synthetic_avg_row, synthetic_hitab_avg_col = compute_avg_rows_cols_for_each_dataset(tables)

    model_name = model.value
    print(f"{tokens_ait_qa} avg tokens are used for AIT-QA dataset to boot-up the graph otter algorithm with {model_name} model")
    #print(f"{ait_avg_row} avg rows and {ait_avg_col} avg cols are used for AIT-QA dataset")


    print(f"{tokens_hitab} avg tokens are used for HITAB dataset to boot-up the graph otter algorithm with {model_name} model")
    #print(f"{hitab_avg_row} avg rows and {hitab_avg_col} avg cols are used for HITAB dataset")

    print(f"{tokens_synthetic} avg tokens are used for Synthetic dataset to boot-up the graph otter algorithm with {model_name} model")
    #print(f"{synthetic_avg_row} avg rows and {synthetic_hitab_avg_col} avg cols are used for Synthetic dataset")
    return

#compute_avg_table_tokens_for_each_model(Models.GEMINI_1_5_FLASH)
#compute_avg_table_tokens_for_each_model(Models.QWEN2)
compute_avg_table_tokens_for_each_model(Models.GEMMA3_4b)