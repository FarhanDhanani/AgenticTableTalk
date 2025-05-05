
from configs import GEMINI_KEYS,hyperparameter, prompt_for_generating_metadata_of_the_markdown_table
from GraphRetriever.graph_retriver import GraphRetriever
from Generator.openai_api import ChatGPTTool
from GraphRetriever.dense_retriever import load_dense_retriever
from Generator.Gemini_model import GeminiTool
# from vertexai.preview import tokenization
import argparse

import tiktoken
from dashscope import get_tokenizer
from iterative_reasoning import GraphReasoner
from compute_score import eval_ex_match,LLM_eval
from configs import DataSet
from tools import table2markdown, table_name_extractor
import json
import jsonlines
import sys
import os
import sys
import select
import time

global tokenizer
logger = None

def str2bool(value):
    if value is None:
        return False
    return str(value).lower() in ("true", "1", "yes", "y")

def augments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=None, required=True)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--key_path", type=str,default=None, required=True)
    parser.add_argument("--dataset", type=str, default=None, required=True)
    parser.add_argument("--qa_path", type=str, required=True)
    parser.add_argument("--table_folder", type=str, required=True)
    parser.add_argument("--eval_model", type=str, required=True)
    parser.add_argument("--eval_model_key_path", type=str, required=True)

    parser.add_argument("--max_iteration_depth", type=int, required=True)
    parser.add_argument("--start", required=False, type=int,default=0)
    parser.add_argument("--end", required=False, type=int,default=None)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--embed_cache_dir", type=str)
    parser.add_argument("--embedder_path", type=str,default='')

    parser.add_argument("--debug", type=str2bool, default=True)
    parser.add_argument("--gpu", type=str2bool, default=False)

    parser.add_argument("--pre_process_tables", type=str2bool, default=False)
    parser.add_argument("--save_markdown", type=str2bool, default=False)
    parser.add_argument("--generate_meta", type=str2bool, default=False)
    parser.add_argument("--exit_after_preprocess", type=str2bool, default=True)
    parser.add_argument("--overwrite_meta_data", type=str2bool, default=False)

    parser.add_argument("--use_heuristic", type=str2bool, default=True)
    args = parser.parse_args()

    return args

def load_model(args):
    if 'gemini' in args.model:
        model = GeminiTool(args)
    else:
        model = ChatGPTTool(args, args.model, args.key_path, args.base_url)

    dense_retriever = load_dense_retriever(args)

    return  model,dense_retriever

def load_eval_model(args):
    model = ChatGPTTool(args, args.eval_model, args.eval_model_key_path)
    return model

# Add the appropriate logic to the load_data function to load 
# the desired dataset from the specified directory.
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



def save_markdown_table(markdown_tables, save_path, file_name, is_save):
    file_location = save_path+file_name+"_markdown.json"
    if is_save:
        with open(file_location, "w") as json_file:
            json.dump(markdown_tables, json_file, indent=4)
    
        print(f"Markdown Tables to {save_path}")
    return file_location

def generate_table_meta(markdown_file, model_name, model, save_path, file_name, is_generate_meta, overwrite_meta):
    meta_file_location = save_path + file_name + "_"+model_name+"_meta.json"
    meta_data = {}

    if os.path.exists(meta_file_location):
        with open(meta_file_location, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
    

    if is_generate_meta:
        with open(markdown_file, 'r', encoding='utf-8') as file:
            markdown_tables = json.load(file)

        if isinstance(markdown_tables, dict):
            try:
                for key, value in markdown_tables.items():
                    table = value
                    if (not overwrite_meta) and meta_data and (key in meta_data):
                        print(key + " already exists")
                    else:
                        prompt = prompt_for_generating_metadata_of_the_markdown_table.format(table=table)
                        result = model.generate(prompt, system_instruction="You are an expert in summarizing Markdown tables.")
                        meta_data[key] = result
            except Exception as e:
                print('encountered an error: ',e.__str__())

            with open(meta_file_location, "w") as json_file:
                json.dump(meta_data, json_file, indent=4)

            print(f"Meta data file is save to {save_path}")
    return meta_file_location

def pre_process_tables(args, model, dense_retriever, querys, answers, table_captions, tables, table_paths):
    if args.pre_process_tables:
        markdown_tables = {}
        for query, answer, caption, table, table_path in zip(querys, answers, table_captions, tables, table_paths):
            graph_retriever = GraphRetriever(table_path, model, dense_retriever, args.dataset, args.use_heuristic, table_cation=caption)
        
            table_name = table_name_extractor(args.dataset, table_path)     
            table = table2markdown(graph_retriever.dealed_rows)

            if table_name not in markdown_tables:
                markdown_tables[table_name] = table
        file_location = save_markdown_table(markdown_tables, args.embed_cache_dir, args.dataset, args.save_markdown)
        meta_file_location = generate_table_meta(file_location, args.model, model, args.embed_cache_dir, 
                                                 args.dataset, args.generate_meta, args.overwrite_meta_data)
        if args.exit_after_preprocess:
            sys.exit(0)
    return

def input_with_timeout(prompt, timeout, default):
    # works for mac/linux only
    print(prompt, end="", flush=True)
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        return sys.stdin.readline().strip().lower()
    return default

def main():
    args = augments()

    model, dense_retriever = load_model(args)
    eval_model = load_eval_model(args)
    querys, answers, table_captions, tables, table_paths = load_data(args)

    pre_process_tables(args, model, dense_retriever, querys, answers, table_captions, tables, table_paths)
    model_name = args.model
    analysis_file_loc =   args.embed_cache_dir + args.dataset+ "analysis" + "_" + model_name.replace("/", "-") + ".json"
    analysis = {}

    if os.path.exists(analysis_file_loc):
        with open(analysis_file_loc, "r", encoding="utf-8") as f:
            analysis = json.load(f)

    total_num,EM,LLM_EVAL = 0,0,0
    start_time = time.time()  # Record the start time
    try:
        for query,answer,caption,table,table_path in zip(querys, answers, table_captions, tables,table_paths):
            table_name = table_name_extractor(args.dataset, table_path)
            if True or ((query + " : " + table_name) not in analysis) or (not analysis[query + " : " + table_name]["output_prediction"]):
                unsafe = False
                error = 3
                graph_retriever = GraphRetriever(table_path, model, dense_retriever, args.dataset, args.use_heuristic, table_cation=caption)
                graph_reasoner = GraphReasoner(args, model, query, table, caption, graph_retriever, table_name, args.dataset)

                #output, iterations = graph_reasoner.iterative_reasoning()
                while error >0:
                    try:
                        output, iterations = graph_reasoner.iterative_reasoning()
                        break
                    except UserWarning as v:
                        print(query+ '\t' + '\t'+args.dataset+ '\t' +'Does not meet security protocol' + v.__str__()) # Did not pass Gemini's security protocol
                        unsafe = True
                        break
                    except Exception as e:
                        print(query+ '\t' +'An error occurred'+ '\t' +e.__str__())
                        error -= 1
                        continue
                if unsafe or error <= 0:
                    #user_input = input("Press 'b' to break or any other key to continue: ").strip().lower()
                    user_input = input_with_timeout("Press 'b' to break or any other key to continue (waiting 20s): ", 20, "continue")
                    if user_input == 'b':
                        break
                    continue

                print("The model's response is: ",output, "\nThe answer is: ",answer)
                print("-------------------------------------------------------------")
                em = eval_ex_match(output,answer)

                if(em):
                    llm_eval = 1
                elif (not em) and (not output) and (answer):
                    llm_eval = 0
                else:
                    llm_eval = LLM_eval(eval_model,query,output,answer)
                    
                analysis[query + " : " + table_name] = {
                    "query" :  query,
                    "table_name" : table_name,
                    "expected_gold_answer": answer,
                    "output_prediction": output,
                    "iterations": iterations,
                    "EM" : em,
                    "LLM_EVAL": llm_eval,
                }

                # Saving the intermidiate analysis
                if (total_num%5 == 0):
                    with open(analysis_file_loc, "w") as json_file:
                        json.dump(analysis, json_file, indent=4)

                # Saving after 5 minutes        
                if time.time() - start_time >= 300:
                    print("\n 5 minutes have passed! Keep going! \n")
                    start_time = time.time()
                    with open(analysis_file_loc, "w") as json_file:
                        json.dump(analysis, json_file, indent=4)
                    
            else:
                em = analysis[query + " : " + table_name]["EM"]
                llm_eval = analysis[query + " : " + table_name]["LLM_EVAL"]

            total_num += 1
            EM += em
            LLM_EVAL += llm_eval
    except Exception as e:
        print('encountered an error: ',e.__str__())

    print('EM:',EM/total_num)
    print('LLM EVAL:',LLM_EVAL/total_num)
    # Saving the final analysis
    with open(analysis_file_loc, "w") as json_file:
        json.dump(analysis, json_file, indent=4)

    return




if __name__ == '__main__':
    main()