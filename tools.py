import logging
from logging.handlers import TimedRotatingFileHandler
import pandas as pd
import os
from configs import DataSet
from enum import Enum
import json

class ModelMetaAlias(Enum):
    OPEN_AI = 'gpt-4o'


# Log
def logger_config(logging_name,logging_path):
    '''
    Configure log
    :param log_path: Output log path
    :param logging_name: Name in the log, can be anything
    :return:
    '''
    '''
        logger is the log object, handler is the stream processor, console is the output to the console 
        (if there is no console, it will not output to the console but will output to the log file)
    '''

    # Get logger object, assign a name
    logger = logging.getLogger(logging_name)
    # Output DEBUG and higher level messages, the first level of filtering for all outputs
    logger.setLevel(level=logging.DEBUG)
    # Get file log handler and set log level, second level of filtering
    # handler = logging.FileHandler(settings.LOG_PATH, encoding='UTF-8')
    handler = TimedRotatingFileHandler(filename=logging_path, when="D", interval=1, backupCount=7,encoding='utf-8')
    handler.setLevel(logging.INFO)
    # Generate and set file log format
    formatter = logging.Formatter("%(asctime)s %(pathname)s %(filename)s %(funcName)s %(lineno)s \
      %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    # console is equivalent to console output, handler is for file output. 
    # Get stream handler and set log level, second level of filtering
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # Add handler to logger object
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def table2markdown(table_list):
    header = table_list[0]
    table_head = '| ' + ' | '.join(header) + ' |' + "\n"  # Concatenate table header
    # Get table body
    table_body = table_list[1:]
    new_table_body = []
    # Convert each list item to a string
    for i in table_body:
        row = []
        for j in i:  # Iterate through this row
            row.append(str(j))  # Convert to string and add to row list
        new_table_body.append(row)  # Then add row to new_table_body
    # Concatenate list body
    table_body = '\n'.join(['| ' + ' | '.join(i) + ' |' for i in new_table_body])
    # Create list separator
    table_split = '| --- ' * len(header) + ' |' + "\n"
    # Concatenate into table variable
    table = table_head + table_split + table_body

    return table

def table2Tuple(table):
    cells = []
    for r in range(len(table)):
        row_tuple = []
        for c in range(len(table[r])):
            row_tuple.append((r,c,table[r][c]))
        row_tuple = '\t'.join([str(i) for i in row_tuple])
        cells.append(row_tuple)
    cells = '\n'.join(cells)
    return str(cells)


def table_name_extractor(dataset, table_path):
    if dataset.lower() in (DataSet.HITAB.value):
        table_name = table_path.split('/')[-1].split('.')[0]   
    elif dataset.lower() in (DataSet.AIT_QA.value):
        table_name = table_path['table_id']
    else:
        raise ValueError('The dataset is not supported')
    return table_name

def get_model_alias_for_pre_processed_meta(model_name):
    model_alias_for_pre_process = ModelMetaAlias.OPEN_AI.value
    
    # if model_name.lower().startswith("gpt"):
    #     model_alias_for_pre_process = ModelMetaAlias.OPEN_AI.value
    # elif "/" in model_name:
    #     model_alias_for_pre_process = model_name.split("/")[-1]  # Extract suffix after '/'
    # else:
    #     model_alias_for_pre_process =  model_name
    
    return model_alias_for_pre_process
    
def read_pre_process_tabular_meta(save_path, file_name, model_name):
    model_alias_for_pre_process = get_model_alias_for_pre_processed_meta(model_name)
    meta_file_location = save_path + file_name + "_" + model_alias_for_pre_process +"_meta.json"
    meta_data = {}

    if os.path.exists(meta_file_location):
        with open(meta_file_location, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
    return meta_data


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # Check if the folder exists, if not, create it as a folder
        os.makedirs(path)  # makedirs will create the path if it does not exist when creating a file
    else:
        pass



