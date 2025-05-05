import os
import json
import jsonlines
import warnings
import tiktoken
from enum import Enum
from transformers import AutoTokenizer
from vertexai.preview import tokenization
from huggingface_hub import login, whoami
from transformers import Qwen2TokenizerFast
from GraphRetriever.graph_retriver import GraphRetriever
from huggingface_hub.utils import RepositoryNotFoundError
from tools import table2Tuple, read_pre_process_tabular_meta
from GraphRetriever.dense_retriever import load_dense_retriever
from GraphRetriever.table_reader import hitab_table_converter,ait_qa_converter

USE_OLLAMA_EMBEDDER = "jmorgan/gte-small:latest"
hugging_face_token_path = "./keys/huggingface_pass.txt"

class DataSet(Enum):
    HITAB = 'hitab'
    AIT_QA = 'ait-qa'

class Models(Enum):
    GEMINI_1_5_FLASH =  "gemini-1.5-flash"
    QWEN2 = 'Qwen/Qwen2-7B-Instruct'
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
       
LLM_select_topk_cell_system_instruction_english = """Suppose you are an expert in statistical analysis.
You will be given a Table described in a special format.
Your task is to identify the cells in the Table that is most relevant to the Question.

Each cell in the Table is represented by a tuple (Row Index, Column Index, Cell Content). 
For example, the tuple (7, 0, "416") represents a cell at row 7, column 0, with a value of "416".
Make sure you read and understand these instructions carefully.
"""

LLM_select_topk_cell_examples = """Here are some examples:
Table Caption: number of internet-enabled devices used per household member by household income quartile, households with home internet access, 2018 
Table:
(0, 0, '')	(0, 1, 'lowest quartile')	(0, 2, 'second quartile')	(0, 3, 'third quartile')	(0, 4, 'highest quartile')	(0, 5, 'total')
(1, 0, '')	(1, 1, 'percent')	(1, 2, 'percent')	(1, 3, 'percent')	(1, 4, 'percent')	(1, 5, 'percent')
(2, 0, 'less than one device per household member')	(2, 1, '63.0')	(2, 2, '60.7')	(2, 3, '56.9')	(2, 4, '56.2')	(2, 5, '58.4')
(3, 0, 'at least one device per household member')	(3, 1, '37.0')	(3, 2, '39.3')	(3, 3, '43.1')	(3, 4, '43.9')	(3, 5, '41.6')

Question: who were less likely to have less than one device per household member,households in the third quartile or those in the lowest quartile?
Select Cell Tuples: [{"tuple": (2, 0, "less than one device per household member"), "explanation": "This row shows the probability of households having less than one device per member, which we need to compare for the third and lowest quartiles."}, {"tuple": (0, 3, "third quartile"), "explanation": "This column represents the third quartile income group, for which we need the probability of having less than one device per member."}, {"tuple": (1, 3, "percent"), "explanation": "This cell confirms that the data in the third quartile column is in percentages, allowing for direct comparison of probabilities."}, {"tuple": (0, 1, "lowest quartile"), "explanation": "This column represents the lowest quartile income group, for which we need the probability of having less than one device per member."}, {"tuple": (1, 1, "percent"), "explanation": "This cell confirms that the data in the lowest quartile column is in percentages, allowing for direct comparison of probabilities."}]

Question: among households in the highest income quartile,what was the percentage of those who had less than one device per household member?
Select Cell Tuples: [{"tuple": (2, 0, "less than one device per household member"), "explanation": "This cell corresponds to the subset of households with less than one device per member, which is what the question asks about."}, {"tuple": (0, 4, "highest quartile"), "explanation": "This cell refers to the highest income quartile, which is the target group in the question."}, {"tuple": (1, 4, "percent"), "explanation": "This cell indicates that the data presented is in percentage format, aligning with how the question is phrased."}]

Question: among households in the lowest income quartile,what was the percentage of those who had less than one internet-enabled device for each household member?
Select Cell Tuples: [{"tuple": (0, 1, "lowest quartile"), "explanation": "This cell identifies the column representing the lowest income quartile, which is the target population of the question."}, {"tuple": (2, 0, "less than one device per household member"), "explanation": "This cell identifies the row corresponding to households with less than one device per member, which is what the question asks about."}, {"tuple": (1, 1, "percent"), "explanation": "This cell indicates that the values in the following rows of this column are percentages, the unit asked for in the question."}]

Question: among households who had internet access at home,what was the percentage of those who had less than one internet-enabled device per household member.
Select Cell Tuples: [{"tuple": (2, 0, "less than one device per household member"), "explanation": "This tuple is the row header for households with less than one internet-enabled device per household member."}, {"tuple": (0, 5, "total"), "explanation": "This tuple is the column header for the total percentage across all income quartiles."}, {"tuple": (1, 5, "percent"), "explanation": "This tuple is the row header for the percentage."}]

Table Caption: mental health indicators, by sexual orientation and gender, canada, 2018 
Table:
(0, 0, 'indicator')	(0, 1, 'heterosexual')	(0, 2, 'heterosexual')	(0, 3, 'heterosexual')	(0, 4, 'gay or lesbian')	(0, 5, 'gay or lesbian')	(0, 6, 'gay or lesbian')	(0, 7, 'bisexual')	(0, 8, 'bisexual')	(0, 9, 'bisexual')	(0, 10, 'sexual orientation n.e.c')	(0, 11, 'sexual orientation n.e.c')	(0, 12, 'sexual orientation n.e.c')	(0, 13, 'total sexual minority')	(0, 14, 'total sexual minority')	(0, 15, 'total sexual minority')
(1, 0, 'indicator')	(1, 1, 'percent')	(1, 2, '95% confidence interval')	(1, 3, '95% confidence interval')	(1, 4, 'percent')	(1, 5, '95% confidence interval')	(1, 6, '95% confidence interval')	(1, 7, 'percent')	(1, 8, '95% confidence interval')	(1, 9, '95% confidence interval')	(1, 10, 'percent')	(1, 11, '95% confidence interval')	(1, 12, '95% confidence interval')	(1, 13, 'percent')	(1, 14, '95% confidence interval')	(1, 15, '95% confidence interval')
(2, 0, 'indicator')	(2, 1, 'percent')	(2, 2, 'from')	(2, 3, 'to')	(2, 4, 'percent')	(2, 5, 'from')	(2, 6, 'to')	(2, 7, 'percent')	(2, 8, 'from')	(2, 9, 'to')	(2, 10, 'percent')	(2, 11, 'from')	(2, 12, 'to')	(2, 13, 'percent')	(2, 14, 'from')	(2, 15, 'to')
(3, 0, 'self-rated mental health')	(3, 1, '')	(3, 2, '')	(3, 3, '')	(3, 4, '')	(3, 5, '')	(3, 6, '')	(3, 7, '')	(3, 8, '')	(3, 9, '')	(3, 10, '')	(3, 11, '')	(3, 12, '')	(3, 13, '')	(3, 14, '')	(3, 15, '')
(4, 0, 'positive')	(4, 1, '88.9')	(4, 2, '88.3')	(4, 3, '89.4')	(4, 4, '80.3')	(4, 5, '75.5')	(4, 6, '84.4')	(4, 7, '58.9')	(4, 8, '52.4')	(4, 9, '65.2')	(4, 10, '54.6')	(4, 11, '37.4')	(4, 12, '70.9')	(4, 13, '67.8')	(4, 14, '63.6')	(4, 15, '71.7')
(5, 0, 'negative')	(5, 1, '10.7')	(5, 2, '10.2')	(5, 3, '11.2')	(5, 4, '19.7')	(5, 5, '15.6')	(5, 6, '24.4')	(5, 7, '40.9')	(5, 8, '34.7')	(5, 9, '47.4')	(5, 10, '45.4')	(5, 11, '29.1')	(5, 12, '62.6')	(5, 13, '32.1')	(5, 14, '28.2')	(5, 15, '36.3')
(6, 0, 'ever seriously contemplated suicide')	(6, 1, '14.9')	(6, 2, '14.3')	(6, 3, '15.5')	(6, 4, '29.9')	(6, 5, '25.1')	(6, 6, '35.1')	(6, 7, '46.3')	(6, 8, '39.8')	(6, 9, '52.8')	(6, 10, '58.7')	(6, 11, '42.1')	(6, 12, '73.5')	(6, 13, '40.1')	(6, 14, '36.1')	(6, 15, '44.3')
(7, 0, 'diagnosed mood or anxiety disorder')	(7, 1, '16.4')	(7, 2, '15.8')	(7, 3, '17.0')	(7, 4, '29.6')	(7, 5, '24.7')	(7, 6, '35.0')	(7, 7, '50.8')	(7, 8, '44.4')	(7, 9, '57.2')	(7, 10, '40.9')	(7, 11, '25.8')	(7, 12, '58.0')	(7, 13, '41.1')	(7, 14, '36.9')	(7, 15, '45.3')
(8, 0, 'mood disorder')	(8, 1, '9.5')	(8, 2, '9.1')	(8, 3, '10.0')	(8, 4, '20.6')	(8, 5, '16.4')	(8, 6, '25.4')	(8, 7, '36.2')	(8, 8, '30.0')	(8, 9, '42.9')	(8, 10, '31.1')	(8, 11, '18.0')	(8, 12, '48.1')	(8, 13, '29.1')	(8, 14, '25.3')	(8, 15, '33.3')
(9, 0, 'anxiety disorder')	(9, 1, '12.5')	(9, 2, '12.0')	(9, 3, '13.1')	(9, 4, '23.4')	(9, 5, '18.8')	(9, 6, '28.8')	(9, 7, '41.6')	(9, 8, '35.5')	(9, 9, '47.9')	(9, 10, '30.4')	(9, 11, '18.5')	(9, 12, '45.8')	(9, 13, '33.1')	(9, 14, '29.2')	(9, 15, '37.2')

Question: how many percent of heterosexual canadians have reported a mood or anxiety disorder diagnosis?
Select Cell Tuples: [{"tuple": (7, 0, "diagnosed mood or anxiety disorder"), "explanation": "This cell identifies the key indicator related to the question, which is the 'diagnosed mood or anxiety disorder'."},{"tuple": (1, 1, "percent"), "explanation": "This cell indicates that the values in Column 1 are presented as percentages, which is the measurement type the question asks for."},{"tuple": (0, 1, "heterosexual"), "explanation": "This cell identifies the demographic group 'heterosexual', which is the focus of the question."}]

Question: how many percent of sexual minority canadians have reported that they had been diagnosed with a mood or anxiety disorder?
Select Cell Tuples: [{"tuple": (7, 0, "diagnosed mood or anxiety disorder"), "explanation": "This cell contains the indicator 'diagnosed mood or anxiety disorder', which is central to the question."},{"tuple": (0, 13, "total sexual minority"), "explanation": "This cell identifies the 'total sexual minority' group, which is the population of interest in the question."},{"tuple": (1, 13, "percent"), "explanation": "This cell indicates that the data in this column is measured in 'percent', which is the specific measure the question asks for."}]

Question: how many percent of bisexual canadians and gay or lesbian canadians, respectively, have reported poor or fair mental health?
Select Cell Tuples: [{"tuple": (0, 7, "bisexual"), "explanation": "This cell indicates the column header related to 'bisexual' sexual orientation, relevant to the question."},{"tuple": (0, 4, "gay or lesbian"), "explanation": "This cell indicates the column header related to 'gay or lesbian' sexual orientation, relevant to the question."},{"tuple": (5, 0, "negative"), "explanation": "This cell indicates a 'negative' mental health status, which corresponds to poor or fair mental health, directly related to the question."},{"tuple": (1, 4, "percent"), "explanation": "This cell indicates that the column contains percentage values, which are necessary to answer the question about the percentage of gay or lesbian Canadians."},{"tuple": (1, 7, "percent"), "explanation": "This cell indicates that the column contains percentage values, which are necessary to answer the question about the percentage of bisexual Canadians."}]
"""

LLM_select_topk_cell_from_table_english = """Let’s think step by step as follows and give full play to your expertise as a statistical analyst: 
1. **Understand the Question**: Clearly understand the Question and the information needed to answer the Question to determine the necessary information to extract. 
2. **Analyze the Data Structure**: Have a comprehensive understanding of the data in the Table, including the meaning, data types, and formats of each cell tuples.    
3. **Select Relevant Data**: Based on the Question, identify the most relevant cell tuples. **Note:** Pay special attention to the header cell tuples in the Table, as they are often more relevant to the Question's semantics and can help in identifying the related evidence cell tuples.

{examples}

{table}

**Question:** {question}

Output format instructions:
1. Outputs cell tuples in descending order of relevance. 
2. Using this JSON schema: Tuple = {"tuple": tuple, "explanation": str}.  Return a `list[Tuple]`. 
"""

prompt_for_generating_metadata_of_the_markdown_table = """
Analyze the following Markdown table and generate a concise metadata or schema representation that describes the structure and semantics of the table. The schema should be in a structured format, such as JSON, and should include the following elements:

Table Name: A name that describes the purpose or context of the table.
Columns: A detailed description of each column, including:
Column Name: The header as written in the table.
Data Type: An inferred data type (e.g., string, number, percentage, etc.) based on the provided data.
Description: A brief explanation of the column's purpose or content.
Rows: An optional concise description of any logical grouping of rows or unique identifiers, if applicable.
Units or Measurement Context: If specific units (e.g., dollars, percentages) are implied or explicit in the data, capture this.
Data Sources or Notes: Any relevant notes or sources mentioned in the table or inferred based on the context.
Anomalies or Missing Data: Identify and note any missing or blank values.
Date Context: Clarify the meaning of the years,months, or dates in the context of the table and how they apply to each of the rows.
Input Table:
{table}

Your output should accurately represent the structure of the table and clarify any implicit meanings or patterns observed in the data. Ensure the representation is concise, machine-readable, and well-organized for easy integration into a data pipeline or analysis framework.
"""


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

    elif model == Models.QWEN2:
        login_hugging_face_with_token(hugging_face_token_path)
        tokenizer = AutoTokenizer.from_pretrained(model.value)

    elif model == Models.CHAT_GPT:
        tokenizer = tiktoken.encoding_for_model(model.value)

    else:
        raise ValueError("Unsupported model type!")
    return tokenizer

def count_tokens(text, tokenizer):
    """Counts the number of tokens in the given text using the provided tokenizer."""
    if isinstance(tokenizer, Qwen2TokenizerFast):  # For Qwen2 and other Hugging Face models
        tokens = tokenizer(text)["input_ids"]
        return len(tokens)
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

def compute_excess_tokens_in_pre_filter_stage(querys, answers, table_captions, tables, table_paths, tokenizer, table_converter):
    total_tokens = 0
    for query, answer, caption, table, table_path in zip(querys, answers, table_captions, tables, table_paths):
        dealed_rows, dealed_cols, rows, cols, merged_cells = table_converter(table_path)
        cells = table2Tuple(dealed_rows)
        if caption:
            table = f"Table Caption: {caption} \n**Table:**\n {str(cells)}"
        else:
            table = f"**Table:**\n{str(cells)}"
        
        prompt = LLM_select_topk_cell_from_table_english.replace('{question}', query).replace('{table}', table).replace('{examples}',LLM_select_topk_cell_examples)
        system_prompt = LLM_select_topk_cell_system_instruction_english
        tokens = count_tokens(text=system_prompt, tokenizer=tokenizer)  + count_tokens(prompt, tokenizer=tokenizer)
        total_tokens += tokens
    
    return total_tokens

def compute_excess_tokens_before_orignal_graph_otter_boot_up(model: Models, start=0, end=None):
    tokenizer = get_tokenizer(model)
    
    
    querys, answers, table_captions, tables, table_paths = load_data(build_args_for_ait_qa(model, start, end))
    tokens_ait_qa = compute_excess_tokens_in_pre_filter_stage(querys, answers, table_captions, tables, table_paths, tokenizer, ait_qa_converter)
    

    querys, answers, table_captions, tables, table_paths = load_data(build_args_for_hitab(model, start, end))
    tokens_hitab = compute_excess_tokens_in_pre_filter_stage(querys, answers, table_captions, tables, table_paths, tokenizer, hitab_table_converter)

    model_name = model.value
    print(f"{tokens_ait_qa} tokens are used for AIT-QA dataset to boot-up the graph otter algorithm with {model_name} model")
    print(f"{tokens_hitab} tokens are used for HITAB dataset to boot-up the graph otter algorithm with {model_name} model")
    print(f"{tokens_ait_qa+tokens_hitab} tokens are used in total to boot-up the graph otter algorithm with {model_name} model")

    return


def compute_table_tokens_in_thought_prompt(querys, answers, table_captions, tables, table_paths, tokenizer, table_converter):
    total_tokens = 0
    for query, answer, caption, table, table_path in zip(querys, answers, table_captions, tables, table_paths):
       
        markdown_table = "**Table:**\n{}".format(table)
       
        if caption:
            table_caption = caption
            table_caption = 'Table Caption: {}\n'.format(table_caption)
            markdown_table = table_caption + markdown_table
       
        tokens = count_tokens(text=markdown_table, tokenizer=tokenizer)
        total_tokens += tokens
    
    return total_tokens

def compute_table_tokens_injected_in_thought_prompt_via_orignal_graph_otter_algo(model: Models, start=0, end=None):
    tokenizer =  get_tokenizer(model)
    
    querys, answers, table_captions, tables, table_paths = load_data(build_args_for_ait_qa(model, start, end))
    tokens_ait_qa = compute_table_tokens_in_thought_prompt(querys, answers, table_captions, tables, table_paths, tokenizer, ait_qa_converter)
    

    querys, answers, table_captions, tables, table_paths = load_data(build_args_for_hitab(model, start, end))
    tokens_hitab = compute_table_tokens_in_thought_prompt(querys, answers, table_captions, tables, table_paths, tokenizer, hitab_table_converter)

    model_name = model.value
    print(f"{tokens_ait_qa} table tokens are used for AIT-QA dataset in thought prompt of the graph otter algorithm with {model_name} model")
    print(f"{tokens_hitab} table tokens are used for HITAB dataset in thought prompt of the graph otter algorithm with {model_name} model")
    print(f"{tokens_ait_qa+tokens_hitab} table tokens are used in total in thought prompt of the graph otter algorithm with {model_name} model")

    return

def compute_tabular_meta_tokens_geneated_via_proposed_heuristic(meta_json:dict, tokenizer):
    total_tokens = 0
    for table_id, table_meta in meta_json.items():
        tokens = count_tokens(text=table_meta, tokenizer=tokenizer)
        total_tokens += tokens
    return total_tokens

def compute_tabular_meta_tokens_injected_in_though_prompt_via_proposed_heuristic(model: Models, start=0, end=None):
    tokenizer = get_tokenizer(model)

    args = build_args_for_ait_qa(model, start, end)
    meta_json = read_pre_process_tabular_meta(args.embed_cache_dir, args.dataset, args.model)
    tokens_ait_qa = compute_tabular_meta_tokens_geneated_via_proposed_heuristic(meta_json, tokenizer)

    args = build_args_for_hitab(model, start, end)
    meta_json = read_pre_process_tabular_meta(args.embed_cache_dir, args.dataset, args.model)
    tokens_hitab = compute_tabular_meta_tokens_geneated_via_proposed_heuristic(meta_json, tokenizer)

    model_name = model.value
    print(f"{tokens_ait_qa} table meta tokens are used for AIT-QA dataset in thought prompt via porposed heuristic with {model_name} model")
    print(f"{tokens_hitab} table meta tokens are used for HITAB dataset in thought prompt via porposed heuristic with {model_name} model")
    print(f"{tokens_ait_qa+tokens_hitab} table meta tokens are used in total in thought prompt via porposed heuristic with {model_name} model")
    return

#TODO
def calculate_average_percent_of_explored_tables_for_given_dataset(args, querys, answers, table_captions, tables, table_paths,
                                                                   min_heuristic_coef, max_heuristic_coef, enforce_heuristic_max=True):
    dense_retriever = load_dense_retriever(args)
    average_percent_of_explored_tables = 0
    average_number_of_K_retrieved_cells = 0
    count_of_processed_instances = 0

    for query, answer, caption, table, table_path in zip(querys, answers, table_captions, tables, table_paths):
        graph_retriever = GraphRetriever(table_path, None, dense_retriever, args.dataset, True,
                                         min_heuristic_coef=min_heuristic_coef,
                                         max_heuristic_coef=max_heuristic_coef,
                                         enforce_heuristic_max=enforce_heuristic_max,
                                         table_cation=caption)
        
        retriver_cell_topk, retriever_match_id_list, _ = graph_retriever.search_cell(query)
        total_cells_in_table = len(graph_retriever.cell_ids)
        retrieved_cells = len(retriever_match_id_list)
        percent_of_explored_tables = retrieved_cells/total_cells_in_table * 100
        average_percent_of_explored_tables += percent_of_explored_tables
        average_number_of_K_retrieved_cells+=retrieved_cells
        count_of_processed_instances+=1

    average_percent_of_explored_tables /= len(querys)
    average_number_of_K_retrieved_cells/=len(querys)
    return average_percent_of_explored_tables, average_number_of_K_retrieved_cells

#TODO
def calculate_average_percent_of_explored_tables():
    args_for_ait_qa = build_args_for_ait_qa(None)
    querys, answers, table_captions, tables, table_paths = load_data(args_for_ait_qa)
    average_percent_of_explored_tables_in_ait_qa_dataset, average_number_of_K_retrieved_cells_ait_qa = \
        calculate_average_percent_of_explored_tables_for_given_dataset(args_for_ait_qa, querys, answers, table_captions, tables, table_paths, 3, 50)
    
    args_for_hitab = build_args_for_hitab(None)
    querys, answers, table_captions, tables, table_paths = load_data(args_for_hitab)
    average_percent_of_explored_tables_in_hitab_dataset, average_number_of_K_retrieved_cells_hitab = \
        calculate_average_percent_of_explored_tables_for_given_dataset(args_for_hitab, querys, answers, table_captions, tables, table_paths, 6, 60)
    
    print(f"Average percent of explored tables in AIT-QA dataset: {average_percent_of_explored_tables_in_ait_qa_dataset}")
    print(f"Average number of K retrieved cells in AIT-QA dataset: {average_number_of_K_retrieved_cells_ait_qa}")
    print(f"Average percent of explored tables in HITAB dataset: {average_percent_of_explored_tables_in_hitab_dataset}")
    print(f"Average number of K retrieved cells in HITAB dataset: {average_number_of_K_retrieved_cells_hitab}")
    return

    
   

    
def calculate_average_percent_of_explored_tables_with_graph_otter_for_given_dataset(args, querys, answers, table_captions, tables, table_paths, num_of_cells_to_select=8):
    dense_retriever = load_dense_retriever(args)
    average_percent_of_explored_tables = 0
    average_number_of_K_retrieved_cells = 0
    count_of_processed_instances = 0

    for query, answer, caption, table, table_path in zip(querys, answers, table_captions, tables, table_paths):
        graph_retriever = GraphRetriever(table_path, None, dense_retriever, args.dataset, True,
                                         table_cation=caption)
        
        
        total_cells_in_table = len(graph_retriever.cell_ids)
        retrieved_cells = num_of_cells_to_select
        percent_of_explored_tables = retrieved_cells/total_cells_in_table * 100
        average_percent_of_explored_tables += percent_of_explored_tables
        average_number_of_K_retrieved_cells+=retrieved_cells
        count_of_processed_instances+=1

    average_percent_of_explored_tables /= len(querys)
    average_number_of_K_retrieved_cells/=len(querys)
    return average_percent_of_explored_tables, average_number_of_K_retrieved_cells

def calculate_average_percent_of_explored_tables_with_graph_otter():
    args_for_ait_qa = build_args_for_ait_qa(None)
    
    querys, answers, table_captions, tables, table_paths = load_data(args_for_ait_qa)
    average_percent_of_explored_tables_in_ait_qa_dataset, average_number_of_K_retrieved_cells_ait_qa = calculate_average_percent_of_explored_tables_with_graph_otter_for_given_dataset(args_for_ait_qa, querys, answers, table_captions, tables, table_paths)
    
    args_for_hitab = build_args_for_hitab(None)
    querys, answers, table_captions, tables, table_paths = load_data(args_for_hitab)
    average_percent_of_explored_tables_in_hitab_dataset, average_number_of_K_retrieved_cells_hitab = calculate_average_percent_of_explored_tables_with_graph_otter_for_given_dataset(args_for_hitab, querys, answers, table_captions, tables, table_paths)

    print(f"Average percent of explored tables in AIT-QA dataset: {average_percent_of_explored_tables_in_ait_qa_dataset}")
    print(f"Average number of K retrieved cells in AIT-QA dataset: {average_number_of_K_retrieved_cells_ait_qa}")
    print(f"Average percent of explored tables in HITAB dataset: {average_percent_of_explored_tables_in_hitab_dataset}")
    print(f"Average number of K retrieved cells in HITAB dataset: {average_number_of_K_retrieved_cells_hitab}")
    return

def calculate_input_tokens_to_generate_table_meta(markdown_file, tokenizer):

    with open(markdown_file, 'r', encoding='utf-8') as file:
        markdown_tables = json.load(file)

    total_tokens = 0
    if isinstance(markdown_tables, dict):
        try:
            for key, value in markdown_tables.items():
                table = value
                prompt = prompt_for_generating_metadata_of_the_markdown_table.format(table=table)
                system_instruction="You are an expert in summarizing Markdown tables."
                tokens = count_tokens(text=system_instruction, tokenizer=tokenizer)  + count_tokens(prompt, tokenizer=tokenizer)
                total_tokens += tokens
                   
        except Exception as e:
            print('encountered an error: ',e.__str__())
       
    return total_tokens

def compute_input_tokens_spent_to_generate_tabular_meta():
    tokenizer = get_tokenizer(Models.CHAT_GPT)
    input_tokens_spent_to_generate_meta_for_ait_qa = calculate_input_tokens_to_generate_table_meta("./dataset/AIT-QA/ait-qa_markdown.json", tokenizer)
    input_tokens_spent_to_generate_meta_for_hitab = calculate_input_tokens_to_generate_table_meta("./dataset/hitab/hitab_gpt-4o_meta.json", tokenizer)

    print(f"Total Input Tokens Spent to generate Meta Data for Tables of AIT-QA Dataset {input_tokens_spent_to_generate_meta_for_ait_qa}")
    print(f"Total Input Tokens Spent to generate Meta Data for Tables of HITAB Dataset {input_tokens_spent_to_generate_meta_for_hitab}")
    return

# COUNT OF TOKENS SPENT TO BOOT-UP THE GRAPH OTTER ALGORITHM WITH INITIAL SUITABLE TABLE CELL WITH RESSPECT TO GIVEN QUERY
#compute_excess_tokens_before_orignal_graph_otter_boot_up(Models.GEMINI_1_5_FLASH)
#compute_excess_tokens_before_orignal_graph_otter_boot_up(Models.QWEN2)


# COUNT OF TABLE TOKENS INJECTED IN THOUGHT PROMPT VIA ORIGNAL GRAPH OTTER ALGORITHM
#compute_table_tokens_injected_in_thought_prompt_via_orignal_graph_otter_algo(Models.GEMINI_1_5_FLASH)
#compute_table_tokens_injected_in_thought_prompt_via_orignal_graph_otter_algo(Models.QWEN2)

# COUNT OF META TOKENS INJECTED IN THOUGHT PROMPT VIA PROPOSED HEURISTIC
#compute_tabular_meta_tokens_injected_in_though_prompt_via_proposed_heuristic(Models.GEMINI_1_5_FLASH)
compute_tabular_meta_tokens_injected_in_though_prompt_via_proposed_heuristic(Models.QWEN2)


# COUNT OF OUTPUT META TOKENS GENERATED VIA GPT-4o
#compute_tabular_meta_tokens_injected_in_though_prompt_via_proposed_heuristic(Models.CHAT_GPT)

# COUNT OF INPUT TOKENS SPENT TO GENERATE META DATA OF THE MARKDOWN TABLE
#compute_input_tokens_spent_to_generate_tabular_meta()

# COUNT AVERAGE PERCENT OF EXPLORED TABLES
#calculate_average_percent_of_explored_tables()

# COUNT AVERAGE PERCENT OF EXPLORED TABLES WITH GRAPH OTTER
#calculate_average_percent_of_explored_tables_with_graph_otter()