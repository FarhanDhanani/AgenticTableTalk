import ast
import os
import re
import json
from Generator.openai_api import ChatGPTTool


def split_checks(input_string):
    """
    This function extracts substrings from the input string that match a specific pattern 
    involving square brackets. It uses regular expressions to find all occurrences of 
    words followed by square-bracketed content.

    Purpose:
    - The function takes an input string and extracts all substrings that match the pattern
      of a word followed by square brackets containing any characters.
    - It returns a list of all matched substrings found in the input string.

    Parameters:
    - input_string: A string in which the function searches for patterns of words followed 
                    by square-bracketed content.

    Returns:
    - A list of strings that match the pattern `word[content]`, where `word` is a sequence 
      of alphanumeric characters and `content` is any substring inside the square brackets.
    - If no matches are found, an empty list is returned.

    Example:
    - For input: `"foo[bar] baz[qux]"`
      The output will be: `['foo[bar]', 'baz[qux]']`
    - For input: `"hello[world] this is a test"`
      The output will be: `['hello[world]']`
    - For input: `"no matches here"`
      The output will be: `[]` (empty list, as no matches are found)

    """
    # pattern = r'[\w]+\(.*?\)'
    pattern = r'[\w]+\[.*?\]'
    # Use re.findall to get all matches
    result = re.findall(pattern, input_string)
    return result

def output_json_parser(input_string):
    """
    This function parses a string that may contain JSON-formatted data, specifically looking 
    for structures related to functions and their explanations. It attempts to extract and 
    return a list of function details and corresponding explanations from the input string.

    Purpose:
    - The function cleans and processes the input string to identify JSON structures embedded 
      within it (such as functions and explanations) and parses them into usable Python data structures.
    - It returns two lists:
      1. A list of function details (specifically 'Function' data).
      2. A list of explanations (formatted string of function details and explanations).

    Parameters:
    - input_string: A string that may contain JSON-formatted content, including functions and explanations.

    Returns:
    - A tuple containing two lists:
      1. `func_list`: A list of 'Function' data extracted from the JSON-like structures in the string.
      2. `explanation_list`: A list of formatted strings with function names and their explanations.
    
    Example:
    - Input:
      ```json
      ```json
      [
          {
              "Function": {
                  "function_name": "myFunction",
                  "parameters": ["param1", "param2"]
              },
              "Explanation": "This is the explanation for myFunction"
          }
      ]
      ```
      ```
    - Output:
      ([{'function_name': 'myFunction', 'parameters': ['param1', 'param2']}], 
       ['Function: myFunction(param1, param2), Explanation: This is the explanation for myFunction'])
    
    If no valid JSON or function data is found, it returns empty lists.

    """
    input_string = input_string.strip().strip('\n').strip().replace('\n','')
    json_match = re.search(r'```json(.+)```', input_string, re.DOTALL)
    json_match2 = re.search(r'\[.+\]', input_string)

    json_match3 = re.search(r'\{.+\}', input_string)

    match_str = []
    if json_match:
        json_string = json_match.group(1)
        match_str = ast.literal_eval(json_string.strip().strip('\n'))
    elif input_string.strip('```').strip('"').strip("'").strip().startswith('[') and input_string.strip(
            '```').strip('"').strip("'").strip().endswith(']'):
        match_str = ast.literal_eval(input_string.strip('```').strip('"').strip("'").strip())
    elif json_match2:
        json_str = json_match2.group()
        try:
            # Convert JSON string to list
            match_str = ast.literal_eval(json_str)
        except Exception as e:
            match_str = detailed_rules_to_extract_json_from_string(json_str) 
    if len(match_str) ==0 or type(match_str[0]) != dict or "Function" not in match_str[0].keys():
        if input_string.strip('```').strip('"').strip("'").strip().startswith('{') and input_string.strip(
                '```').strip('"').strip("'").strip().endswith('}'):
            try:            
                match_str = ast.literal_eval(input_string.strip('```').strip('"').strip("'").strip())
                match_str = [match_str]
            except Exception as e:
                match_str = detailed_rules_to_extract_json_from_string(match_str)
                match_str = [match_str]
        elif json_match3:
            json_str = json_match3.group()
            try:
                # Convert JSON string to list
                match_str = ast.literal_eval(json_str)
                match_str = [match_str]
            except Exception as e:
                match_str = detailed_rules_to_extract_json_from_string(json_str)
                match_str = [match_str]
        else:
            return [],[]


    func_list = []
    exlanation_lsit = []
    for match in match_str:
        try:
            func_list.append(match['Function'])
            exlanation_lsit.append(
                f"Function: {match['Function']['function_name']}({', '.join([str(i) for i in match['Function']['parameters']])}), Explanation: {match['Explanation']}")
        except KeyError as k:
            try:
                list_of_matches = extract_nested_functions(match)
                for m in list_of_matches:
                    func_list.append(m['Function'])
                    exlanation_lsit.append(
                        f"Function: {m['Function']['function_name']}({', '.join([str(i) for i in m['Function']['parameters']])}), Explanation: {m['Explanation']}")
            except Exception as e:
                print('LLM action output parsing error',k.__str__())
                raise Exception(f'LLM action output parsing error {k.__str__()}')
            # continue
    return func_list, exlanation_lsit

def get_action_list(string, model):
    # if string[:len('Finish')] == 'Finish':
    #     return [string]
    # else:
    #     # return string.split(', ')
    # return split_checks(string)
    try:
        actionlist = output_json_parser(string)
    except (SyntaxError, ValueError) as e:
        improvedJsonString = json_output_parser_llm_powered(string, model)
        actionlist = output_json_parser(improvedJsonString)
    return actionlist



def parse_action_json(function_dict):
    """
    This function parses a dictionary representing a function call in JSON format, extracting the function name
    and its parameters. It processes the parameters to handle different formats such as tuples, lists, and strings 
    with special formatting. The function then returns the parsed function name and a list of processed parameters.

    Purpose:
    - The function extracts the 'function_name' and 'parameters' from the given `function_dict`.
    - It handles complex parameter formats, such as lists of tuples or strings that represent tuples, and ensures 
      proper parsing using `ast.literal_eval` for safe evaluation.
    - The function supports handling nested lists and special string patterns (e.g., strings starting and ending with 
      parentheses and containing commas).
    - It returns the function name and the list of processed parameters in a format that can be used by other parts of 
      the system.

    Parameters:
    - function_dict: A dictionary that contains the function name (`'function_name'`) and its associated parameters 
                     (`'parameters'`), where the parameters can be in different formats (list, tuple, string, etc.).

    Outputs:
    - action_type: The function name extracted from `function_dict['function_name']`.
    - parameters: A list of processed parameters. Each parameter is parsed based on its format, such as converting 
                  strings representing tuples into actual tuples, and handling nested lists appropriately.

    Explanation:
    - The function iterates through the items in the `parameters` field, checking the type and structure of each item.
    - If an item is a list that contains tuples or strings resembling tuples, it attempts to parse those elements into 
      appropriate types using `ast.literal_eval` or custom parsing logic for strings.
    - If the item is a list with elements containing the word 'answer' in the function name (case-insensitive), the 
      function adds the elements directly to the `parameters` list.
    - The function also handles string parameters that represent tuples, splitting and converting them into a tuple format 
      as needed.
    - Finally, the function returns the `action_type` (function name) and the fully processed `parameters` list.

    """
    action_type = function_dict['function_name']
    argument = function_dict['parameters']
    parameters = []
    for item in argument:
        if type(item) == list and len(item) > 0 and (type(item[0]) == tuple or (type(item[0]) == str and item[0].startswith('(') and item[0].endswith(')') and item[0].count(',') >= 2)):
            for it in item:
                parameters.append(ast.literal_eval(it) if type(it) == str and it.startswith('(') and it.endswith(')') and it.count(',') >= 2 else it )
        elif type(item) == list and 'answer' in (action_type.lower()):
            for it in item:
                parameters.append(it)
        elif type(item) == list:
            parameters.append(tuple(item))
        elif type(item) == str and  item.startswith('(') and item.endswith(')') and item.count(',') >= 2:
            try:
                parameters.append(ast.literal_eval(item))
            except Exception as e:
                print('LLM output action cannot be parsed',item,e.__str__())
                item_list = [i.strip().strip('(').strip(')').strip('"').strip("'").strip() for i in item.split(',')]
                parameters.append((int(item_list[0]),int(item_list[1]),''.join(item_list[2:])))
        else:
            parameters.append(item)
    return action_type,parameters

def LLM_json_output_parse(output):
    """
    This function parses the output from an LLM (Large Language Model) that is expected to contain a JSON 
    string within it. It uses regular expressions to identify and extract the JSON data, cleans up the 
    string, and attempts to convert it into a Python dictionary or list. If parsing fails, it raises 
    a UserWarning with an error message.

    Purpose:
    - The function takes an output string that may contain JSON data embedded within it.
    - It uses regular expressions to extract JSON data in different formats (such as wrapped in ```json ... ``` or 
      directly as a JSON object or array).
    - It then tries to parse the JSON string into a Python object using `ast.literal_eval`.
    - If successful, it returns the parsed result; otherwise, it raises a `UserWarning`.

    Parameters:
    - output: A string representing the LLM output that may contain JSON data.

    Outputs:
    - result: The parsed Python object (e.g., dictionary or list) extracted from the JSON string.
    - If parsing fails, a UserWarning is raised with details about the error.

    Explanation:
    - The function first strips unnecessary characters (like newline or extra spaces) from the `output`.
    - It searches for JSON data within the output using regular expressions (`re.search`).
    - If the output contains JSON wrapped in ```json ... ```, it extracts the content, cleans it up, and attempts to parse it.
    - If the output is directly a valid JSON object or array (either starting and ending with `{}` or `[]`), it tries to parse it.
    - It also handles cases where the output contains standalone JSON-like strings and tries to extract and convert them.
    - If no valid JSON is found, the function raises a `UserWarning` with an error message indicating that the parsing failed.

    Example:
    If the `output` is:
    ```
    ```json
    {"key": "value"}
    ```
    ```
    The function will extract the JSON string `{"key": "value"}` and convert it to a Python dictionary.

    If no valid JSON is found, it will raise a `UserWarning` indicating the error.

    """
  
    output = output.strip().strip('\n').strip().replace('\n','')
    json_match = re.search(r'```json(.+)```', output, re.DOTALL)
    # Use regular expressions to extract JSON
    json_pattern = r'\{.+\}'
    json_match2 = re.search(json_pattern, output)
    json_match3 = re.search(r'\[.+\]', output)
    if json_match:
        json_string = json_match.group(1)
        try:
            result = ast.literal_eval(json_string.strip().strip('\n'))
        except Exception as e:
            print('LLM output parsing error', output, e.__str__())
            result = detailed_rules_to_extract_json_from_string(json_string.strip().strip('\n'))
    elif output.strip('```').strip('"').strip("'").strip().startswith('{') and output.strip(
            '```').strip('"').strip("'").strip().endswith('}'):
        try:
            result = ast.literal_eval(output.strip('```').strip('"').strip("'").strip())
        except Exception as e:
            print('LLM output parsing error', output, e.__str__())
            result = detailed_rules_to_extract_json_from_string(output.strip('```').strip('"').strip("'").strip())
    elif output.strip('```').strip('"').strip("'").strip().startswith('[') and output.strip(
            '```').strip('"').strip("'").strip().endswith(']'):
        try:
            result = ast.literal_eval(output.strip('```').strip('"').strip("'").strip())
        except Exception as e:
            print('LLM output parsing error', output, e.__str__())
            result = detailed_rules_to_extract_json_from_string(output.strip('```').strip('"').strip("'").strip())
    elif json_match2:
        try:
            json_str = json_match2.group()
            # Convert JSON string to dictionary
            result = ast.literal_eval(json_str)
        except Exception as e:
            print('LLM output parsing error', output, e.__str__())
            result = detailed_rules_to_extract_json_from_string(json_str)
    elif json_match3:
        try:
            json_str = json_match3.group()
            # Convert JSON string to dictionary
            result = ast.literal_eval(json_str)
        except Exception as e:
            print('LLM output parsing error', output, e.__str__())
            result = detailed_rules_to_extract_json_from_string(json_str)
    else:
        raise UserWarning('LLM output parsing error ' + output)        

    return result


def remove_quotes(s):
    """
    This function removes leading and trailing quotes from a given string. It handles various types of 
    quotes, including single quotes ('), double quotes ("), and curly quotes (e.g., ‘, ’, “, ”).

    Purpose:
    - The function takes a string `s` as input and removes any leading and trailing quotes, whether 
      they are standard single/double quotes or curly quotes commonly used in typographic text.
    - It returns the modified string without the surrounding quotes if they exist.

    Parameters:
    - s: A string that may contain quotes at the beginning and/or end.

    Returns:
    - A string with any leading and trailing quotes removed.
    - If no quotes are found at the beginning or end of the string, the original string is returned unchanged.

    Example:
    - For input: `"hello"`
      The output will be: `hello`
    - For input: `'world'`
      The output will be: `world`
    - For input: `‘text’`
      The output will be: `text`

    """
    s = s.strip().strip('\n')
    if s.startswith(("'", '"','‘','’','“','”')):
        s = s[1:]
    if s.endswith(("'", '"','‘','’','“','”')):
        s = s[:-1]
    return s


def json_output_parser_llm_powered(json_string, model):
    parsing_prompt = f"""The following input may contain malformed or incomplete JSON. Please correct its syntax and structure by referring to the example format below. Return only the corrected JSON array — no explanation or extra text.

Example Format:
[
    {{
        "Function": {{
            "function_name": "myFunction",
            "parameters": ["param1", "param2"]
        }},
        "Explanation": "This is the explanation for myFunction"
    }}
]

Input JSON to correct:
{json_string}
"""
    model = ChatGPTTool(None, "gpt-4o-mini-2024-07-18", "keys/openai_pass.txt")

    result = model.generate(parsing_prompt,
                            system_instruction="You are a JSON syntax corrector. Your task is to fix malformed or improperly structured JSON based on a given example format. Ensure the corrected output strictly follows the structure shown in the example, including key names, array usage, and formatting. Return only the corrected JSON array with no additional commentary or explanation."
                            )
  
    return result

def detailed_rules_to_extract_json_from_string(json_string):
  first_brace = min(
      (json_string.find('[') if '[' in json_string else float('inf')),
      (json_string.find('{') if '{' in json_string else float('inf'))
  )
  
  last_brace = max(
      json_string.rfind(']'),
      json_string.rfind('}')
      )
  # Extract the substring that should contain valid JSON
  cleaned = json_string[first_brace:last_brace + 1]
  
  # Try parsing it
  return ast.literal_eval(cleaned)


def extract_nested_functions(obj):
    results = []

    def find_deep_function(o):
        if isinstance(o, dict):
            if 'function_name' in o and 'parameters' in o:
                return o
            for value in o.values():
                result = find_deep_function(value)
                if result:
                    return result
        elif isinstance(o, (list, tuple)):
            for item in o:
                result = find_deep_function(item)
                if result:
                    return result
        return None

    def recurse(o):
        if isinstance(o, dict):
            if 'Function' in o and 'Explanation' in o:
                deep_func = find_deep_function(o['Function'])
                if deep_func:
                    results.append({
                        'Function': deep_func,
                        'Explanation': o['Explanation']
                    })
            for value in o.values():
                recurse(value)
        elif isinstance(o, (list, tuple)):
            for item in o:
                recurse(item)

    recurse(obj)
    return results
