import json
import jsonlines
from enum import Enum
base_path = "./dataset/"
file_names = ["gemini-1.5-flash.json", "qwen-qwen-2-72b-instruct.json", "qwen-qwen3-14b.json", "qwen-qwen3-8b.json", "google-gemma-3-12b-it.json", "google-gemma-3-4b-it.json"];
#file_names = ["qwen-qwen3-14b.json"]
class DataSet(Enum):
    HITAB = 'hitab'
    AIT_QA = 'AIT-QA'
    SYNTHETIC = 'synthetic'

def table_name_extractor(dataset, table_path):
    if dataset.lower() in (DataSet.HITAB.value.lower()):
        table_name = table_path['table_id']
    elif dataset.lower() in (DataSet.AIT_QA.value.lower()):
        table_name = table_path['table_id']
    elif dataset.lower() in (DataSet.SYNTHETIC.value.lower()):
        table_name = table_path['table_id']
    else:
        raise ValueError('The dataset is not supported')
    return table_name

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load the JSON file
    return data

def analyze_predictions(pediction_json_path):
    data = read_json_file(pediction_json_path)
    total_entries = len(data.items())
    empty_count = sum(1 for key, value in data.items() if value.get("output_prediction") in [None, ""])
    correct_count_llm = sum(1 for key, value in data.items() if value.get("LLM_EVAL") in[1])
    correct_count_em = sum(1 for key, value in data.items() if value.get("EM") in[1])
    average_iterations = sum(value.get("iterations") for key, value in data.items()) / total_entries
    return empty_count, total_entries, correct_count_llm, correct_count_em, average_iterations

def count_of_unique_table_ids_in_input_queries(dataset, query_jsonl_path):
    qas = []
    table_ids = set()
    if dataset.lower() in (DataSet.HITAB.value.lower()):
        with open(query_jsonl_path, "r+", encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                qas.append(item)
    elif dataset.lower() in (DataSet.AIT_QA.value.lower()):
        qas = read_json_file(query_jsonl_path)
    
    elif dataset.lower() in (DataSet.SYNTHETIC.value.lower()):
        qas = read_json_file(query_jsonl_path)

    for item in qas:
        table_name = table_name_extractor(dataset, item)
        if table_name not in table_ids:
            table_ids.add(table_name)
    
    return len(table_ids)

def count_of_total_number_of_queries(datset, query_jsonl_path, log=False):
    qas = []
    if datset.lower() in (DataSet.HITAB.value.lower()):
        with open(query_jsonl_path, "r+", encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                qas.append(item)
            
    elif datset.lower() in (DataSet.AIT_QA.value.lower()):
        qas = read_json_file(query_jsonl_path)

    elif datset.lower() in (DataSet.SYNTHETIC.value.lower()):
        qas = read_json_file(query_jsonl_path)
    
    else:
        raise ValueError('The dataset is not supported')
    
    unique_queries = {}
    for item in qas:
        table_name = table_name_extractor(datset, item)
        query = item['question']
        key = query + " : " + table_name
        unique_queries[key] = unique_queries.get(key, 0) + 1
    
    duplicates = {key: count for key, count in unique_queries.items() if count > 1}
    if (log):
        print(f"{datset} :duplpicates are: \n")
        for key, count in duplicates.items():
            print(f"Query: {key} Count: {count}")
    return len(qas), len(unique_queries), len(duplicates)

def analyze_meta_file(meta_json_path):
    data = read_json_file(meta_json_path)
    total_entries = len(data)
    return total_entries


def describe_ait_qa_experiment():
    ait_qa_base_path = base_path + DataSet.AIT_QA.value + "/"
    input_file_loc = ait_qa_base_path+"aitqa_clean_questions.json"
    total_queries_count, unique_queries_count, duplicate_queries_count = count_of_total_number_of_queries(DataSet.AIT_QA.value, input_file_loc)
    unique_table_ref_count = count_of_unique_table_ids_in_input_queries(DataSet.AIT_QA.value, input_file_loc)

    meta_file_path = ait_qa_base_path+"ait-qa_gpt-4o_meta.json"
    total_meta_entries = analyze_meta_file(meta_file_path)

    print(f"***************----------- (AIT-QA Dataset Analysis) ----------*********************")
    print(f"Total Number of inputs queries: {total_queries_count}")
    print(f"Total Number of duplicate queries: {duplicate_queries_count}")
    print(f"Total Number of unique queries: {unique_queries_count}")
    print(f"Total unique table references in input queries: {unique_table_ref_count}")

    print(f"Total entries in meta file: {total_meta_entries}")


    for file_name in file_names:
        pred_json_file_path = ait_qa_base_path+"ait-qaanalysis_"+file_name
        empty_count, total_entries, correct_count_llm, correct_count_em, average_iterations = analyze_predictions(pred_json_file_path)

        print(file_name+" RESULTS: ")
        print(f"    Total Number of Pedictions: {total_entries}")
        print(f"    Average Number of Iterations: {average_iterations}")
        print(f"    Correct Prediction Count (LLM EVAL): {correct_count_llm}")
        print(f"    Correct Prediction Count (EM EVAL): {correct_count_em}")
        print(f"    Total entries with empty 'output_prediction': {empty_count}")

        print(f"    Total Accuracy with (LLM EVAL)': {correct_count_llm/total_entries*100}%")
        print(f"    Total Accuracy with (EM EVAL)': {correct_count_em/total_entries*100}%")
    return

def describe_hitab_experiment():
    hitab_base_path = base_path + DataSet.HITAB.value + "/"
    input_file_loc = hitab_base_path+"test_samples.jsonl"
    total_queries_count, unique_queries_count, duplicate_queries_count = count_of_total_number_of_queries(DataSet.HITAB.value, input_file_loc)
    unique_table_ref_count = count_of_unique_table_ids_in_input_queries(DataSet.HITAB.value, input_file_loc)

    meta_file_path = hitab_base_path+"hitab_gpt-4o_meta.json"
    total_meta_entries = analyze_meta_file(meta_file_path)

    print(f"***************----------- (HITAB Dataset Analysis) ----------*********************")
    print(f"Total Number of inputs queries: {total_queries_count}")
    print(f"Total Number of duplicate queries: {duplicate_queries_count}")
    print(f"Total Number of unique queries: {unique_queries_count}")
    print(f"Total unique table references in input queries: {unique_table_ref_count}")

    print(f"Total entries in meta file: {total_meta_entries}")


    for file_name in file_names:
        pred_json_file_path = hitab_base_path+"hitabanalysis_"+file_name #"hitabanalysis_gemini-1.5-flash.json"
        empty_count, total_entries, correct_count_llm, correct_count_em, average_iterations = analyze_predictions(pred_json_file_path)

        print(file_name+" RESULTS: ")
        print(f"    Total Number of Pedictions: {total_entries}")
        print(f"    Average Number of Iterations: {average_iterations}")
        print(f"    Correct Prediction Count (LLM EVAL): {correct_count_llm}")
        print(f"    Correct Prediction Count (EM EVAL): {correct_count_em}")
        print(f"    Total entries with empty 'output_prediction': {empty_count}")
        
        print(f"    Total Accuracy with (LLM EVAL)': {correct_count_llm/total_entries*100}%")
        print(f"    Total Accuracy with (EM EVAL)': {correct_count_em/total_entries*100}%")
    return


def describe_sd_experiment():
    ait_qa_base_path = base_path + DataSet.SYNTHETIC.value + "/"
    input_file_loc = ait_qa_base_path + "up_generated_large_table_questions.json"
    total_queries_count, unique_queries_count, duplicate_queries_count = count_of_total_number_of_queries(DataSet.SYNTHETIC.value, input_file_loc)
    unique_table_ref_count = count_of_unique_table_ids_in_input_queries(DataSet.SYNTHETIC.value, input_file_loc)

    meta_file_path = ait_qa_base_path+"ait-qa_gpt-4o_meta.json"
    total_meta_entries = analyze_meta_file(meta_file_path)

    print(f"***************----------- (SYNTHETIC Dataset Analysis) ----------*********************")
    print(f"Total Number of inputs queries: {total_queries_count}")
    print(f"Total Number of duplicate queries: {duplicate_queries_count}")
    print(f"Total Number of unique queries: {unique_queries_count}")
    print(f"Total unique table references in input queries: {unique_table_ref_count}")

    print(f"Total entries in meta file: {total_meta_entries}")


    for file_name in file_names:
        pred_json_file_path = ait_qa_base_path+"ait-qaanalysis_"+file_name
        empty_count, total_entries, correct_count_llm, correct_count_em, average_iterations = analyze_predictions(pred_json_file_path)

        print(file_name+" RESULTS: ")
        print(f"    Total Number of Pedictions: {total_entries}")
        print(f"    Average Number of Iterations: {average_iterations}")
        print(f"    Correct Prediction Count (LLM EVAL): {correct_count_llm}")
        print(f"    Correct Prediction Count (EM EVAL): {correct_count_em}")
        print(f"    Total entries with empty 'output_prediction': {empty_count}")

        print(f"    Total Accuracy with (LLM EVAL)': {correct_count_llm/total_entries*100}%")
        print(f"    Total Accuracy with (EM EVAL)': {correct_count_em/total_entries*100}%")
    return

#describe_ait_qa_experiment()
#describe_hitab_experiment()
describe_sd_experiment()
# hitab_base_path = base_path + DataSet.HITAB.value + "/"
# input_file_loc = hitab_base_path+"test_samples.jsonl"
# count_of_total_number_of_queries(DataSet.HITAB.value, input_file_loc)