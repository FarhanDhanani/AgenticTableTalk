import json
import tiktoken

model_to_tokenize = "gpt-3.5-turbo"
detailed_prompt = """
Analyze the following Markdown table and generate a concise metadata or schema representation that describes the structure and semantics of the table. The schema should be in a structured format, such as JSON, and should include the following elements:

Table Name: A name that describes the purpose or context of the table.
Columns: A detailed description of each column, including:
Column Name: The header as written in the table.
Data Type: An inferred data type (e.g., string, number, percentage, etc.) based on the provided data.
Description: A brief explanation of the column's purpose or content.
Rows: An optional description of any logical grouping of rows or unique identifiers, if applicable.
Units or Measurement Context: If specific units (e.g., dollars, percentages) are implied or explicit in the data, capture this.
Data Sources or Notes: Any relevant notes or sources mentioned in the table or inferred based on the context.
Anomalies or Missing Data: Identify and note any missing or blank values.
Date Context: Clarify the meaning of the years,months, or dates in the context of the table and how they apply to each of the rows.
Input Table:
{table}

Your output should accurately represent the structure of the table and clarify any implicit meanings or patterns observed in the data. Ensure the representation is concise, machine-readable, and well-organized for easy integration into a data pipeline or analysis framework.
"""

base_dataset_dir = "./dataset/"
ait_qa_data_path = base_dataset_dir+"AIT-QA/"
hitab_data_path = base_dataset_dir+"hitab/"

ait_qa_file_name = "ait-qa_markdown.json"
hitab_file_name = "hitab_markdown.json"

def count_items(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
        if isinstance(data, dict):
            return len(data)
        else:
            raise ValueError("JSON content is not a dictionary")

def count_tokens_in_prompt(prompt: str, model: str = "gpt-4o") -> int:
    """Counts the number of tokens in a given prompt using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(prompt)
        return len(tokens)
    except Exception as e:
        print(f"Error: {e}")
        return -1

def count_tokens_in_dataset(json_file, model) -> int:
    total_tokens = 0
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
        if isinstance(data, dict):
            for key, value in data.items():
                table = data[key]
                prompt = detailed_prompt.format(table=table)
                total_tokens+=count_tokens_in_prompt(prompt, model=model)
        else:
            raise ValueError("JSON content is not a dictionary")
    return total_tokens

# Example usage
ait_qa_json_file = ait_qa_data_path+ait_qa_file_name  # Replace with your JSON file path
hitab_json_file = hitab_data_path+hitab_file_name  # Replace with your JSON file path
try:
    item_count_ait_qa = count_items(ait_qa_json_file)
    print(f"Number of items in the AIT-QA JSON dictionary: {item_count_ait_qa}")
    tokens_in_ait_qa = count_tokens_in_dataset(ait_qa_json_file, model_to_tokenize)
    print(f"Total number of tokens in the AIT-QA dataset: {tokens_in_ait_qa}")

    item_count_hitab = count_items(hitab_json_file)
    print(f"Number of items in the HITAB JSON dictionary: {item_count_hitab}")
    tokens_in_hitab = count_tokens_in_dataset(hitab_json_file, model_to_tokenize)
    print(f"Total number of tokens in the HITAB dataset: {tokens_in_hitab}")

    print(f"Total numbeer items in both JSON dictionaries: {item_count_hitab+item_count_ait_qa}")
    print(f"Total number of tokens in both datasets: {tokens_in_ait_qa+tokens_in_hitab}")

except Exception as e:
    print(f"Error: {e}")
