from GraphRetriever.table_reader import hitab_table_converter,ait_qa_converter
from collections import OrderedDict

from GraphRetriever.BM25_Retriever import EM_retriever
from configs import PROMPT_TEMPLATE,hyperparameter
import ast
import copy

import os
from Generator.parse_output import remove_quotes
import time
import re
import random
from tools import table2Tuple

from configs import DataSet

# Add appropriate dataset parser to the table_loader_dict dictionary.
# The defination of the dataset parser function should be in the table_reader.py file and imported correctly.
# The key should be the dataset name and the value should be the corresponding parser function.
# Example: 'dataset_name': dataset_parser_function

table_loader_dict = {
    'hitab': hitab_table_converter,
    'ait-qa':ait_qa_converter,

}

class GraphRetriever:
    """
    The GraphRetriever class is designed to load a table, convert it into a graph structure, and perform various 
    retrieval operations on the graph. It uses both sparse and dense retrieval methods to find relevant cells 
    in the table based on a query.

    Instance Variables:
    - dealed_rows: List of processed rows from the table.
    - dealed_cols: List of processed columns from the table.
    - rows: Original rows from the table.
    - cols: Original columns from the table.
    - merged_cells: List of merged cells in the table.
    - table_cation: Caption of the table.
    - cells: List of all cells in the table as strings.
    - dealed_cells: List of all processed cells in the table as strings.
    - cell_ids: List of cell IDs.
    - LLM_model: The language model used for generating prompts and responses.
    - dataset: The dataset name.
    - dense_retriever: The dense retriever object used for dense retrieval operations.
    - use_heuristic: A variable to confirm heurisitic modified implementation has to be run.
    - min_heuristic_coef: A variable to controll minimum number of cells to be retrieved from the dense retriever.
    - max_heuristic_coef: A variable to controll maximum number of cells to be retrieved from the dense retriever.
    - enforce_heuristic_max: Boolean which ensures that max_heuristic_coef is respected while retrieving the cells.
    """

    def __init__(self,table_path,model,dense_retriever,dataset, use_heuristic, min_heuristic_coef=3,
                  max_heuristic_coef=50, enforce_heuristic_max=True,table_cation='', is_rep=0.0):

        self.dealed_rows, self.dealed_cols, \
                self.rows,self.cols,self.merged_cells = table_loader_dict[dataset](table_path)
        self.table_cation = table_cation
        self.cells = [str(c) for r in self.rows for c in r]
        self.dealed_cells = [str(c) for r in self.dealed_rows for c in r]
        self.cell_ids = [i for i in range(len(self.rows)*len(self.rows[0]))]

        self.LLM_model = model

        self.dataset =dataset
        self.dense_retriever = dense_retriever
        self.use_heuristic = use_heuristic
        self.min_heuristic_coef = min_heuristic_coef
        self.max_heuristic_coef = max_heuristic_coef
        self.enforce_heuristic_max = enforce_heuristic_max

        self.dense_retriever.load_graph(self.cells,dataset, 
                                        self.get_file_name(dataset,table_path))
        self.is_rep = is_rep
        return

    def get_file_name(self, dataset, table_path):
        """
        Extracts and returns the appropriate file name based on the dataset type and table path.

        Args:
            dataset (str): The dataset identifier, expected to match values from the `DataSet` enumeration.
            table_path (Union[dict, str]): The table path or metadata, which is used to determine the file name.
                - If the dataset is `AIT_QA`, this should be a dictionary containing a 'table_id' key.
                - If the dataset is `HITAB`, this should be a file path string.

        Returns:
            str: The extracted file name corresponding to the dataset and table path.

        Raises:
            ValueError: If the provided dataset is not supported.

        Purpose:
            - For `AIT_QA` datasets: Returns the `table_id` from the provided dictionary.
            Example:
                dataset = "AIT_QA"
                table_path = {"table_id": "table_123"}
                Result: "table_123"

            - For `HITAB` datasets: Extracts the base name (without extension) from the provided file path.
            Example:
                dataset = "HITAB"
                table_path = "/path/to/table_file.xlsx"
                Result: "table_file"

            - Raises an exception for unsupported datasets, ensuring that only valid datasets are processed.
        """
        file_name = ""
        if dataset in (DataSet.AIT_QA.value):
            file_name = table_path['table_id']
        elif dataset in (DataSet.HITAB.value):
            file_name = os.path.split(os.path.splitext(table_path)[0])[1]
        else:
            raise ValueError(f"Dataset {dataset} not supported")
        return file_name

    def get_neighbors(self,add_id_list,get_same_row,get_all_nei=False):
        """
        Retrieves the neighboring cells (same row, same column, or both) for a given list of cell IDs.

        Args:
            add_id_list (list): A list of cell IDs for which neighbors need to be identified.
            get_same_row (bool): Whether to include cells in the same row as the given cell IDs.
            get_all_nei (bool, optional): If True, includes both same-row and same-column neighbors. Defaults to False.

        Returns:
            tuple:
                - cell_topk (list): The content of the matched cells corresponding to `add_id_list`.
                - result (str): A formatted string summarizing the neighbors in the same row and/or column.
                - hit_cell_neighbors_content_id (OrderedDict): A dictionary mapping neighbor cell IDs to their content.

        Purpose:
            This function identifies neighboring cells in a table based on their position relative to a given cell.
            It allows for flexible retrieval of neighbors in the same row, same column, or both, depending on input flags.

        Example:
            Input:
                add_id_list = ["cell_1"]
                get_same_row = True
                get_all_nei = True
            Output:
                cell_topk = ["content of cell_1"]
                result = "The row containing 'cell_1' includes the following nodes: [...].\n
                        The column containing 'cell_1' includes the following nodes: [...]."
                hit_cell_neighbors_content_id = OrderedDict({
                    "neighbor_cell_id_1": "content of neighbor_cell_id_1",
                    "neighbor_cell_id_2": "content of neighbor_cell_id_2",
                    ...
                })
        """

        # Initialize variables for matched cells and their neighbors
        match_id_list = add_id_list
        cell_topk, _ = self.tableId2Content(add_id_list, self.dealed_rows)

        # Get the cells in the same row/column as the matched cell
        hit_cell_same_row_col = self.getSameRow_ColCells(match_id_list)

        # Retrieve same-row and same-column cell IDs for the matched cell
        same_col_cells_id = hit_cell_same_row_col[add_id_list[0]]['same_col_cells_id']
        same_row_cells_id = hit_cell_same_row_col[add_id_list[0]]['same_row_cells_id']

        # Initialize containers for neighbor data
        hit_cell_neighbors = {
            '同行': [],
            '同列': []
        }
        hit_cell_neighbors_content_id = OrderedDict()

        # Process same-row neighbors if requested
        if get_same_row or get_all_nei:
            for row in same_row_cells_id:
                same_row_cells, same_row_cell_ids = self.tableId2Content(row, self.dealed_rows)
                same_row_cells_tuple = self.cell2Tuple(same_row_cell_ids)
                hit_cell_neighbors['同行'] += same_row_cells_tuple
                for k in range(len(same_row_cells)):
                    hit_cell_neighbors_content_id[same_row_cell_ids[k]] = same_row_cells[k]

        # Process same-column neighbors if requested
        if get_all_nei or not get_same_row:
            for col in same_col_cells_id:
                same_col_cells, same_col_cell_ids = self.tableId2Content(col, self.dealed_cols)
                same_col_cells_tuple = self.cell2Tuple( same_col_cell_ids)
                hit_cell_neighbors['同列']+= same_col_cells_tuple
                for k in range(len(same_col_cells)):
                    hit_cell_neighbors_content_id[same_col_cell_ids[k]] = same_col_cells[k]
        
        # Format the result summary for row and column neighbors
        result = []
        cell_topk_tuple = self.cell2Tuple(match_id_list)
        if get_same_row or get_all_nei:
            result.append(
                    'The row containing "{}" includes the following nodes: {}.'.format(cell_topk_tuple[0] if len(cell_topk_tuple) == 1 else cell_topk_tuple,
                                                                             str(hit_cell_neighbors['同行'])))
        if get_all_nei or not get_same_row:
            result.append(
                'The column containing "{}" includes the following nodes: {}.'.format(cell_topk_tuple[0] if len(cell_topk_tuple) == 1 else cell_topk_tuple,
                                                                                str(hit_cell_neighbors['同列'])))
            
        # Combine results into a single formatted string      
        result = '\n'.join(result)

        # Return matched cells, result summary, and detailed neighbor content
        return cell_topk,result, hit_cell_neighbors_content_id

    def search_cell(self,query,topk=3,LLM_match_id_list=[],select_cell=False):
        """
        This function searches for cells within a table that best match a given query string.
        It combines results from sparse and dense retrieval methods and optionally integrates LLM-based matches. 
        It returns the top-k matching cells, their corresponding IDs, and a tuple representation.

        Args:
            query (str): The query string to search for in the table cells.
            topk (int, optional): The maximum number of top matching results to return. Defaults to 3.
            LLM_match_id_list (list, optional): A list of cell IDs already matched by an LLM for exclusion or prioritization. Defaults to an empty list.
            select_cell (bool, optional): Flag to determine whether only dense retrieval results should be considered. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - cell_topk (list): Top-k matching cell contents.
                - match_id_list (list): Corresponding IDs of the matching cells.
                - cell_topk_tuple (tuple): A structured tuple representation of the matching cells.
        """

        # Retrieve LLM-matched cells based on provided IDs.
        LLM_match_topk = [self.dealed_cells[i] for i in LLM_match_id_list]

        # If the query matches directly with any cell in the table.
        if query in self.cells:
            cell_topk, match_id_list, sparse_scores = EM_retriever(self.dealed_cells, query,isEM=True)
            cell_topk,match_id_list,cell_topk_tuple = self.cell2Tuple(match_id_list,add_merged_cells=True)
            return cell_topk,match_id_list,cell_topk_tuple
        else:
            if not select_cell:
                # Perform sparse retrieval when dense retrieval isn't enforced exclusively.
                sparse_cell_topk, sparse_match_id_list, sparse_scores = EM_retriever(self.dealed_cells, query,isEM=True)
            else:
                # sparse_cell_topk, sparse_match_id_list, sparse_scores = EM_retriever(self.dealed_cells, query)
                sparse_cell_topk, sparse_match_id_list, sparse_scores = [], [], None
            # TODO: check returns
            # Perform dense retrieval using a dense retriever.
            if self.use_heuristic:
                min_threshold_k=self.min_heuristic_coef
                max_threshold_k=self.max_heuristic_coef
                dense_cell_topk, dense_match_id_list, dense_scores = self.dense_retriever.search_with_dynamic_k_and_silhoutee_score(query, min_threshold_k, max_threshold_k)
                    
                if select_cell:
                    neighbours = self.getSameRow_ColCells(dense_match_id_list)
                    respect_max_threshold = self.enforce_heuristic_max
                    for key, value in neighbours.items():
                        for row_index, row_cell_id in enumerate(value['same_row_cells_id'][0]):
                            if (row_cell_id not in dense_match_id_list) and (
                                not respect_max_threshold or 
                                (len(dense_match_id_list) < max_threshold_k and len(dense_cell_topk) < max_threshold_k)
                                ):
                                dense_match_id_list.append(row_cell_id)
                                dense_cell_topk.append(value['same_row_cells'][0][row_index])
                        for col_index, col_cell_id in enumerate(value['same_col_cells_id'][0]):
                            if col_cell_id  not in dense_match_id_list and (
                                not respect_max_threshold or 
                                (len(dense_match_id_list) < max_threshold_k and len(dense_cell_topk) < max_threshold_k)
                                ):
                                dense_match_id_list.append(col_cell_id)
                                dense_cell_topk.append(value['same_col_cells'][0][col_index])
                    
                    cell_topk_tuple = self.cell2Tuple(dense_match_id_list)
                    return dense_cell_topk, dense_match_id_list, cell_topk_tuple
                else:
                    cell_topk,match_id_list,cell_topk_tuple = self.cell2Tuple(dense_match_id_list,add_merged_cells=True)
                    return cell_topk, match_id_list, cell_topk_tuple
            else:
                dense_cell_topk, dense_match_id_list, dense_scores = self.dense_retriever.search_single(query, topk=20)

            if not select_cell:
                # Combine sparse and dense results for final top-k matches.
                cell_topk = sparse_cell_topk
                match_id_list = sparse_match_id_list

                # Augment sparse results with dense matches if necessary.
                if len(cell_topk) < topk:
                    for j in range(len(dense_cell_topk[:(topk- len(cell_topk))])):
                        # Fixed added over here dense_scores[:(topk- len(cell_topk))][j] was producing IndexError: list index out of range error
                        if dense_scores[j] > hyperparameter[self.dataset]['dense_score']:
                            dense_cell  = dense_cell_topk[j]
                            temp_cell_topk,temp_match_id,_ = EM_retriever(self.dealed_cells, dense_cell,isEM=True)
                            cell_topk += temp_cell_topk
                            match_id_list += temp_match_id
                cell_topk,match_id_list,cell_topk_tuple = self.cell2Tuple(match_id_list,add_merged_cells=True)
            else:
                # Filter dense matches by excluding overlaps with sparse and LLM matches.
                dense_topk = topk
                temp_dense_topk,temp_dense_id_list = [],[]
                for i in range(len(dense_match_id_list)):
                    if dense_topk <= 0:
                        break
                    dense_id = dense_match_id_list[i]
                    if dense_id in sparse_match_id_list or dense_id in LLM_match_id_list or dense_cell_topk[i] in LLM_match_topk:
                        dense_topk -= 1
                    else:
                        temp_dense_topk.append(dense_cell_topk[i])
                        temp_dense_id_list.append(dense_match_id_list[i])
                        dense_topk -= 1

                # Filter sparse matches to exclude LLM matches.
                temp,temp_ = [],[]
                for i in range(len(sparse_match_id_list)):
                    if sparse_match_id_list[i] not in LLM_match_id_list:
                        temp.append(sparse_match_id_list[i])
                        temp_.append(sparse_cell_topk[i])
                sparse_match_id_list,sparse_cell_topk = temp,temp_

                # Combine sparse and dense matches into a hybrid result.
                temp = sparse_match_id_list + temp_dense_id_list
                hybird_cell_id = OrderedDict() # Ordered dict contain {cell_id:cell_content, ...}
                hybird_cell = sparse_cell_topk + temp_dense_topk
                for i in range(len(temp)):
                    # if hybird_cell[i]:
                    hybird_cell_id[temp[i]] = hybird_cell[i]
                
                # LLM-based re-ranking or direct extraction of results.
                # cell_topk,match_id_list = self.LLM_reranker(query,hybird_cell_id ,topk)
                cell_topk, match_id_list = list(hybird_cell_id.values()), list(hybird_cell_id.keys())
                cell_topk_tuple = self.cell2Tuple(match_id_list)
            return cell_topk,match_id_list,cell_topk_tuple


    def LLM_generate(self,prompt,system_instruction,response_mime_type,isrepeated=None):
        """
        This function interacts with the LLM model to generate a response based on the provided prompt and system instructions.
        It processes the model's response, extracts JSON data, and returns the parsed result.

        Parameters:
        - prompt (str): The input query or prompt for the LLM model.
        - system_instruction (str): Instructions provided to guide the model's behavior.
        - response_mime_type (str): The format in which the model's response is expected (e.g., JSON).
        - isrepeated (float): A parameter indicating whether the response should repeat. Default is 0.0.

        Returns:
        - select_id (list or dict): The parsed JSON data from the LLM's response, returned as a list or dictionary depending on the output format.
        
        Process:
        1. The function starts by preparing the prompt for the LLM model, adjusting it if the model's name does not contain 'gemini'.
        2. A loop is used to attempt parsing the model's output multiple times (up to 3 attempts).
        3. The function sends the prompt and system instruction to the LLM model for generation.
        4. The response is checked to see if it contains valid JSON. If valid JSON is found, it is parsed into a Python object.
        5. If no valid JSON is found, the function retries the generation process with adjusted parameters (increasing `isrepeated_` and decreasing error attempts).
        6. If the response cannot be parsed after 3 attempts, a warning is raised indicating that the LLM couldn't extract the cell.

        Example:
        If the prompt is "Get information about cell A1", the model might respond with a JSON string like:
        ```
        ```json
        {"cell": "A1", "value": 100}
        ```
        The function will parse this into a dictionary: {"cell": "A1", "value": 100}.
        
        Error Handling:
        - If the model's response cannot be parsed as JSON, the function will retry up to 3 times.
        - If parsing fails after 3 attempts, a warning will be raised: "LLM cannot extract the cell".
        """
        if isrepeated is None:
            isrepeated = self.is_rep
        isrepeated_,error = copy.deepcopy(isrepeated),3
        if 'gemini' not in self.LLM_model.model_name:
            prompt = [prompt]
        while error > 0:
            select_id_text = self.LLM_model.generate(prompt, system_instruction=system_instruction,
                                                response_mime_type=response_mime_type, isrepeated=isrepeated_)
            
            # Look for JSON data within the response
            # select_id_text = select_id['text'].strip().strip('\n').strip()
            json_match = re.search(r'```json(.*?)```', select_id_text, re.DOTALL)
            json_match3 = re.search(r'\[.*\]', select_id_text)
            if json_match:
                # Extract JSON string from the response
                json_string = json_match.group(1)
                try:
                    select_id = ast.literal_eval(json_string.strip().strip('\n'))
                except Exception as e:
                    print('LLM output parsing error',select_id,e.__str__())
                    raise Exception('LLM output parsing error '+ select_id,e.__str__())
            
            # Handle cases where the response is wrapped in square brackets (potential list)
            elif select_id_text.strip('```').strip('"').strip("'").strip().startswith('[') \
                    and select_id_text.strip('```').strip('"').strip("'").strip().endswith(']'):
                try:
                    select_id = ast.literal_eval(select_id_text.strip('```').strip('"').strip("'").strip())
                except Exception as e:
                    print('LLM output parsing error', select_id, e.__str__())
                    raise Exception('LLM output parsing error ' + select_id, e.__str__())
                
            # Check for a valid JSON array at the start or end of the response
            elif json_match3:
                try:
                    json_str = json_match3.group()
                    # Convert JSON string to dictionary
                    select_id = ast.literal_eval(json_str)
                except Exception as e:
                    print('LLM output parsing error', select_id, e.__str__())
                    raise Exception('LLM output parsing error ' + select_id, e.__str__())
            else:
                # Retry generation if no valid JSON is found
                isrepeated_ = 0.7
                error -= 1
                continue
            # if type(select_id) == list and len(select_id) >= 0:
            #     break

            # If the parsed result is a valid list, exit the loop
            if type(select_id) == list and len(select_id) >= 0:
                break
            else:
                # Retry with adjusted parameters
                isrepeated_ = 0.7
                error -= 1
                continue
        
        # Raise warning if parsing fails after multiple attempts
        if error <= 0:
            raise UserWarning('LLM cannot extract the cell')
        
        # Return the parsed result
        return select_id
    
    def initialize_subgraph(self, query):
        """
        # Function: initialize_subgraph
        # Purpose: 
        # This function is designed to initialize the subgraph by selecting relevant cells from the table based on a query.
        # It performs the following steps:
        # 1. It first uses an LLM (Language Model) to select the top-k relevant cells from the table based on the query.
        # 2. Then, it uses a retriever to find additional relevant cells from the table, based on the cells selected by the LLM.
        # 3. After selecting the cells, it identifies cells that are in the same row or column as the matched cells and forms a connected graph based on these neighboring cells.
        # 4. It then stores the cell content and their corresponding IDs for both LLM and retriever selections.
        # 5. Returns a subgraph consisting of relevant cells, their connections, and the resulting data from both the LLM and retriever selection methods.

        # Outputs:
        # 1. cell_id_content_topk: A dictionary containing two keys ('LLM_select' and 'retriever_select'). 
        #    Each key maps to another dictionary that stores the selected cells and their corresponding content.
        # 2. cell_tuple_topk: A dictionary with two keys ('LLM_select' and 'retriever_select'), 
        #    where each key maps to the tuple representation of the selected cells.
        # 3. connect_id_cell_dict: An ordered dictionary that maps cell IDs to their content in the table.
        # 4. connect_graph_result: A string that describes the connected graph of selected cells and their relationships.
        """
        
        if self.use_heuristic:
            LLM_cell_topk = []
            LLM_match_id_list = []
        else:        
            # -LLM select first
            LLM_cell_topk, LLM_match_id_list = self.LLM_select_cells_from_table(query, hyperparameter[self.dataset][
                'LLM_select_cells'])
        
        # TODO 2. Instead of using dense retriever to find relevant cells here we can from the sub query based on the retreived result from above
        # then retriever select
        retriver_cell_topk, retriever_match_id_list, _ = self.search_cell(query, topk=hyperparameter[self.dataset]['dense_topk'],
                                                                          LLM_match_id_list=LLM_match_id_list,
                                                                          select_cell=True)

        cell_topk,match_id_list = LLM_cell_topk + retriver_cell_topk, LLM_match_id_list + retriever_match_id_list
        # Get the cells in the same row/column as the matched cell
        hit_cell_same_row_col = self.getSameRow_ColCells(match_id_list)
        # Get the connected subgraph based on neighbors
        connect_graph, connect_id_cell_dict, connect_graph_result = self.hit_cell_connect_graph(hit_cell_same_row_col)
        cell_id_content_topk = OrderedDict(
            {
                'LLM_select':{},
                'retriever_select': {}
            }
        )
        cell_tuple_topk = copy.deepcopy(cell_id_content_topk)

        for i in range(len(LLM_cell_topk)):
            cell_id_content_topk['LLM_select'][LLM_match_id_list[i]] = LLM_cell_topk[i]
        cell_tuple_topk['LLM_select'] = self.cell2Tuple(LLM_match_id_list)
        for i in range(len(retriver_cell_topk)):
            cell_id_content_topk['retriever_select'][retriever_match_id_list[i]] = retriver_cell_topk[i]
        cell_tuple_topk['retriever_select'] = self.cell2Tuple(retriever_match_id_list)



        return cell_id_content_topk, cell_tuple_topk, connect_id_cell_dict, connect_graph_result



    def getSameRow_ColCells(self,match_cell_id):
        """
        This function identifies the cells in the same row and column as the specified cells in a table. 
        It returns their corresponding cell IDs and content in a structured format.

        Args:
            match_cell_id (list): A list of cell IDs for which the same row and column information is to be determined.

        Returns:
            OrderedDict: A dictionary where each key corresponds to a cell ID from `match_cell_id`, 
                        and the value is a dictionary containing:
                        - 'same_row_cells': List of contents of cells in the same row to the corresponding cell Id present in Key.
                        - 'same_row_cells_id': List of IDs for the cells in the same row to the corresponding cell whose cell Id present in Key.
                        - 'same_col_cells': List of contents of cells in the same column as the key cell to the corresponding cell Id present in Key.
                        - 'same_col_cells_id': List of IDs for the cells in the same column to the corresponding cell whose cell Id present in Key.
        """

        # Number of columns in the table.
        cols_num = len(self.cols)

        # Map to store row and column IDs for each cell in match_cell_id.
        hit_cell_row_col_id = OrderedDict()
        for i in range(len(match_cell_id)):
            # result = topk_result[i]

            # Calculate row index.
            row_id = match_cell_id[i] // cols_num
            # Calculate column index.
            col_id = match_cell_id[i] % cols_num
            hit_cell_row_col_id[match_cell_id[i]] = {
                'row_id': [row_id],
                'col_id': [col_id]
            }

        # Map to store cells in the same row and column for each cell in match_cell_id.
        hit_cell_same_row_col = OrderedDict()
        for key,value in hit_cell_row_col_id.items():
            row_id_list = value['row_id']
            col_id_list = value['col_id']
            hit_cell_same_row_col[key] = {
                # Contents of cells in the same row.# Contents of cells in the same row.
                'same_row_cells': [],
                # IDs of cells in the same row.
                'same_row_cells_id': [],
                # Contents of cells in the same column.
                'same_col_cells': [],
                 # IDs of cells in the same column.
                'same_col_cells_id': []
            }

             # Collect all cells in the same row.
            for r_id in row_id_list:
                row_id_list = [i for i in range(r_id*len(self.rows[0]), (r_id+1)*len(self.rows[0]))]
                hit_cell_same_row_col[key]['same_row_cells'].append(self.dealed_rows[r_id])
                hit_cell_same_row_col[key]['same_row_cells_id'].append(row_id_list)

            # Collect all cells in the same column.
            for c_id in col_id_list:
                col_is_list = [i for i in range(c_id,len(self.rows[0]) *(len(self.cols[c_id]) -1)+ c_id +1,len(self.rows[0]) )]
                hit_cell_same_row_col[key]['same_col_cells'].append(self.dealed_cols[c_id])
                hit_cell_same_row_col[key]['same_col_cells_id'].append(col_is_list)


        return hit_cell_same_row_col


    def hit_cell_connect_graph(self,order_info,get_shared_nei=False):
        """
        Creates a graph connecting cells that share the same row or column based on the provided order_info.

        Purpose:
            This function processes the given `order_info`, which contains information about cells in the same row 
            and column, and constructs a graph that links cells based on their positions. The function can generate 
            detailed information about shared neighboring cells and handle connections between cells within the 
            same row or column. The output is useful for visualizing the relationship and structure of the cells in 
            the table.

        Args:
            order_info (dict): A dictionary containing information about cells in the same row or column, 
                                with keys representing cell identifiers and values representing cell data such as 
                                'same_row_cells_id' and 'same_col_cells_id'.
            get_shared_nei (bool): A flag indicating whether to include shared neighboring cells in the result. 
                                Defaults to False.

        Returns:
            tuple:
                - `connect_graph` (OrderedDict): A dictionary representing the connections between cells, where 
                                                keys are pairs of cells (as strings) and values are lists of 
                                                connected cells.
                - `connect_id_cell_dict` (OrderedDict): A dictionary mapping cell IDs to their respective contents.
                - `connect_graph_result` (str): A formatted string summarizing the graph of connected cells, including 
                                                shared neighboring cells (if `get_shared_nei` is True).

        Notes:
            - The function constructs connections based on the provided information about cells in the same row or 
            column.
            - Duplicates in the connection graph are removed, and the result is sorted.
            - The function also handles situations where no shared neighbors exist, providing customized messages 
            depending on the scenario.
            - If `get_shared_nei` is True, the function will output connections between cells that share neighbors.
            - The final result is a string representation of the connection graph, along with dictionaries containing 
            the mapping of cell IDs and their respective contents.

        Example:
            - If `order_info` contains information about cells in the same row or column, the function will create 
            a connection graph showing which cells are linked, and provide details about the relationships.
        """
        # Initialize graph and lists
        # connect_graph = []
        connect_id_graph = []
        # order_info = hit_cell_same_row_col
        keys = list(order_info.keys())         # keys,_ = self.tableId2Content(keys)

        # Process each key and its associated cell data
        for key,value in order_info.items():
            row_cells_id_list =  value['same_row_cells_id']
            col_cells_id_list = value['same_col_cells_id']

            # Get the index of the current key
            key_index = keys.index(key)
            # key_id = cell_id_list[key_index]

            # Iterate through the rest of the keys to find matching connections
            for index in range(key_index+1 , len(keys)):
                n_key = keys[index]

                n_row_cells_id_list = order_info[n_key]['same_row_cells_id']
                n_col_cells_id_list = order_info[n_key]['same_col_cells_id']

                # Check for connections in the same row
                for row_cells_id in row_cells_id_list:
                    for i in range(len(row_cells_id)):
                        row_id = row_cells_id[i]
                        for n_row_cells_id in n_row_cells_id_list:
                            if row_cells_id == n_row_cells_id:
                                connect_id_graph.append([key, 'SAME ROW', n_key])
                            else:
                                for k in range(len(n_col_cells_id_list)):
                                    n_col_cells_id = n_col_cells_id_list[k]
                                    if row_id in n_col_cells_id:
                                        link_cell_id = n_col_cells_id.index(row_id)
                                        if key != n_col_cells_id_list[k][link_cell_id] and n_col_cells_id_list[k][link_cell_id] != n_key:
                                            connect_id_graph.append([key, 'SAME ROW', n_col_cells_id_list[k][link_cell_id], 'SAME COLUMN', n_key])
                
                # Check for connections in the same column
                for col_cells_id in col_cells_id_list:
                    for i in range(len(col_cells_id)):
                        col_id = col_cells_id[i]
                        for n_col_cells_id in n_col_cells_id_list:
                            if col_cells_id == n_col_cells_id:
                                connect_id_graph.append([key, 'SAME COLUMN', n_key])
                            else:
                                for k in range(len(n_row_cells_id_list)):
                                    n_row_cells_id = n_row_cells_id_list[k]
                                    if col_id in n_row_cells_id:
                                        link_cell_id = n_row_cells_id.index(col_id)
                                        if key != n_row_cells_id_list[k][link_cell_id] and n_row_cells_id_list[k][link_cell_id] != n_key :
                                            connect_id_graph.append(
                                                    [key, 'SAME COLUMN', n_row_cells_id_list[k][link_cell_id], 'SAME ROW', n_key])

        # Remove duplicates from the graph
        connect_id_graph = [tuple(i) for i in connect_id_graph if len(i) > 0 ]
        connect_id_graph = list(set(connect_id_graph))
        connect_id_graph = [list(i) for i in connect_id_graph]

        # Initialize the neighbors and same-row/column list
        connect_graph_neighbors = OrderedDict()
        same_row_col_list = []
        row_width = len(self.rows[0])

        # Process the connections between cells
        for i in range(len(connect_id_graph)):
            id_item = connect_id_graph[i]
            if len(id_item) > 3:
                pair_cell = [id_item[0],id_item[4]]
                if str(pair_cell)  in connect_graph_neighbors.keys():
                    continue
                connect_graph_neighbors[str(pair_cell)] = [id_item[2]]
                for j in range(i + 1, len(connect_id_graph)):
                    next_id_item = connect_id_graph[j]
                    if len(next_id_item) > 3:
                        next_pair_cell = [next_id_item[0], next_id_item[4]]
                        if next_pair_cell == pair_cell:
                            connect_graph_neighbors[str(pair_cell)].append(next_id_item[2])
                connect_graph_neighbors[str(pair_cell)] = sorted(connect_graph_neighbors[str(pair_cell)])
            else:
                connect_graph_neighbors[str(id_item)] = id_item[1]
                same_row_col_list.append(id_item)

        # Further process the connections into tuple format and prepare the final results
        connect_graph_tuples = OrderedDict()
        connect_graph = OrderedDict()
        connect_id_cell_dict = OrderedDict()
        for key,value in connect_graph_neighbors.items():
            pair_cell = ast.literal_eval(key)
            if len(pair_cell) == 2:
                temp = []
                tuple_temp = self.cell2Tuple(pair_cell)
                for cell in pair_cell:
                    row_id = cell // row_width
                    col_id = cell % row_width
                    connect_id_cell_dict[cell] = self.dealed_rows[row_id][col_id]
                    temp.append(self.dealed_rows[row_id][col_id])
                n_temp = []
                tuple_n_temp = self.cell2Tuple(value)
                for cell in value:
                    row_id = cell // row_width
                    col_id = cell % row_width
                    connect_id_cell_dict[cell] = self.dealed_rows[row_id][col_id]
                    n_temp.append(self.dealed_rows[row_id][col_id])
                connect_graph[str(temp)] = n_temp
                connect_graph_tuples[str(tuple_temp)] = tuple_n_temp


        # Final result: Prepare the connect graph result as a string
        connect_graph_result = []
        contents,_ = self.tableId2Content(keys,self.dealed_rows)
        for key_item in contents: # Output in the order of the searched cells.
            for key, value in connect_graph_tuples.items():
                temp_key = [key[2] for key in ast.literal_eval(key)]
                if temp_key[0] == key_item:
                    if type(value) == list:
                        connect_graph_result.append(
                            PROMPT_TEMPLATE[self.dataset]['shared_neighbors'].replace('{cell_pair}',key )
                            .replace(
                                '{shared_cells}', str(connect_graph_tuples[key])))

        # Organize the same-row/column result if necessary
        same_row_col_result,same_row_col_tuple_result = self.organize_same_row_col(same_row_col_list, keys)
        if not get_shared_nei:
            connect_graph_result += same_row_col_tuple_result
            # connect_graph_result = same_row_col_tuple_result + connect_graph_result

        # Handle cases with no shared neighbors
        tuple_contents = self.cell2Tuple(keys)
        connect_graph_result = sorted(set(connect_graph_result), key=connect_graph_result.index)
        if len(connect_graph_result) == 0 and get_shared_nei:
            if len(same_row_col_result) > 0:
                if "SAME ROW" in same_row_col_result[0]:
                    connect_graph_result = PROMPT_TEMPLATE[self.dataset]['no_shared_neighbors_but_same_row'].replace(
                        '{cell_pair}', str(tuple_contents))
                else:
                    connect_graph_result = PROMPT_TEMPLATE[self.dataset]['no_shared_neighbors_but_same_col'].replace(
                        '{cell_pair}', str(tuple_contents))
            else:
                connect_graph_result = PROMPT_TEMPLATE[self.dataset]['no_shared_neighbors'].replace('{cell_pair}',
                                                                                                    str(tuple_contents))
        else:
            connect_graph_result = '\n'.join(connect_graph_result)


        return connect_graph,connect_id_cell_dict,connect_graph_result

    def tableId2Content(self,id_list, table,hit_cell_id=None):
        """
        This function retrieves the content and corresponding IDs of cells in a table based on their indices and 
        optionally checks for overlapping merged cells with a specified cell.

        Args:
            id_list (list):
                A list of cell IDs for which content is to be retrieved. 
                IDs are represented as integer indices based on table dimensions.

            table (list of lists): 
                A 2D representation of the table where each element corresponds to a cell's content.

            hit_cell_id (int, optional): 
                The ID of a specific cell to check against merged cell ranges. 
                Defaults to None, meaning no merged cell check is performed.

        Returns:
            tuple:
                - content_list (list): A list of cell contents corresponding to the given IDs.
                - content_id_list (list): A list of cell IDs for which content was successfully retrieved.
        """
        # Stores the content of the cells based on `id_list`.
        content_list = []
        # Stores the IDs of the cells whose content is added to `content_list`.
        content_id_list = []
        # Tracks whether a cell is part of a merged region overlapping with `hit_cell_id`.
        is_hit = False
        for id in id_list:
            # Calculate the row and column indices of the current cell.
            row_id = id // len(table[0])
            col_id = id % len(table[0])

            # Retrieve the content of the cell at the current row and column.
            content = table[row_id][col_id]

            if hit_cell_id != None:
                # Calculate the row and column indices of the cell to check against merged cells.
                hit_cell_row_id = hit_cell_id // len(table[0])
                hit_cell_col_id = hit_cell_id % len(table[0])
                for rlo, rhi, clo, chi in self.merged_cells:
                    # Check if the current cell and the hit cell belong to the same merged cell range.
                    if row_id in range(rlo, rhi) and col_id in range(clo, chi) and  hit_cell_row_id in range(rlo, rhi) and hit_cell_col_id in range(clo,chi):
                        is_hit = True
                        break # Exit the loop as we found an overlapping merged region.
            if is_hit:
                # Reset the flag for the next iteration.
                is_hit = False
                # Skip adding this cell to the result lists.
                continue
            
            # Add the cell's content and ID to the respective lists.
            # if content:
            content_id_list.append(id)
            content_list.append(content)
        # Return the content and IDs of the valid cells.
        return content_list,content_id_list

    def LLM_select_cells_from_table(self, query, topk=3,prompt_tamplate= 'LLM_select_cells'):

        """
        This function generates a prompt for the LLM model to select specific cells from a table based on a given query. 
        It formats the input query and table data, sends the prompt to the LLM model, and then processes the model's output 
        to return the selected cells from the table.

        Parameters:
        - query (str): The question or query that specifies the cells to select from the table.
        - topk (int): The number of top cells to retrieve from the LLM model. Default is 3.
        - prompt_tamplate (str): The template used to generate the prompt for the LLM model. Default is 'LLM_select_cells'.

        Returns:
        - result (list): A list of cell contents corresponding to the selected cell IDs.
        - select_id_list (list): A list of cell IDs corresponding to the selected cells.

        Process:
        1. The function starts by converting the table data into a tuple format (cell addresses).
        2. If a table caption exists, it adds it to the table information.
        3. It constructs a prompt using the provided query, topk value, table data, and other information (such as system instructions and examples).
        4. The prompt is sent to the LLM model for cell selection using the `LLM_generate` method.
        5. Once the model returns the selected cells, the function processes the response:
            - The model's output is parsed to extract the cell tuples (row, column, content).
            - The function cleans up and formats the selected cell tuples.
            - It calculates the cell IDs based on the row and column indices.
        6. The cell IDs are used to retrieve the corresponding cell contents from the table.
        7. The function returns the selected cell contents and their corresponding cell IDs.

        Example:
        If the input query is "Select the values from rows 2 and 3," the function will prompt the LLM to select cells based on this query, 
        process the model's response, and return the corresponding cell contents and their IDs.

        Error Handling:
        - If the model's response contains invalid or unparseable cell tuples, the function attempts to clean the data and retry parsing.

        """

        # Convert the table into a tuple format and prepare the table caption if provided
        cells = table2Tuple(self.dealed_rows)
        if self.table_cation:
            table = f"Table Caption: {self.table_cation} \n**Table:**\n {str(cells)}"
        else:
            table = f"**Table:**\n{str(cells)}"

        # Prepare system instruction and examples for the prompt
        # No topK variable was present in the prompt
        system_instruction = PROMPT_TEMPLATE[self.dataset]['LLM_select_cells_system_instruction'].replace('{topk}', str(topk))
        examples = PROMPT_TEMPLATE[self.dataset]['LLM_select_cell_examples']

        # Generate the final prompt using the input query, table, and other relevant information
        prompt = PROMPT_TEMPLATE[self.dataset][prompt_tamplate]\
            .replace('{question}', query).replace('{table}', table).replace('{topk}', str(topk)).replace('{examples}',examples)
        
        # Print the prompt for debugging if in debug mode
        if self.LLM_model.args.debug:
            print('prompt for selecting a cell in a large mode: ',prompt)
        # dialogs = [
        #     {"role": "user", "content": prompt}
        # ]

        # Generate response using LLM model
        select_id = self.LLM_generate(prompt,system_instruction=system_instruction,
                                      response_mime_type="application/json" if 'gemini' in self.LLM_model.model_name else {"type": "json_schema"})
        print('the cell initially selected by the model is: ', select_id)
        # select_id = select_id['text'].split('\n')

        # Initialize list to hold valid cell IDs
        select_id_list = []

        # Process the selected cells (topk limit)
        for i in range(len(select_id)):
            if i >= topk:
                break
            cell_tuple_ = select_id[i]
            cell_tuple = cell_tuple_['tuple']

            # Clean and format the cell tuple if it's in string format
            cell_tuple = remove_quotes(cell_tuple).strip(',').strip('.').strip('，').strip('。').strip() if type(cell_tuple) == str else cell_tuple
            if type(cell_tuple) == str and cell_tuple.startswith('(') and cell_tuple.endswith(')'):
                # cell_tuple = ast.literal_eval(cell_tuple)
                try:
                    cell_tuple = ast.literal_eval(cell_tuple)
                except Exception as e:
                    print('LLM output action cannot be parsed', cell_tuple, e.__str__())
                    item_list = [i.strip().strip('(').strip(')').strip('"').strip("'").strip() for i in cell_tuple.split(',')]
                    cell_tuple = (int(item_list[0]), int(item_list[1]), ''.join(item_list[2:]))
            
            # Calculate the cell ID based on row and column indices
            row_num = int(cell_tuple[0])
            col_num = int(cell_tuple[1])
            cell_id = len(self.dealed_rows[0]) * row_num + col_num

            # Check if the cell ID is valid and append to the list
            if cell_id in self.cell_ids:
                select_id_list.append(cell_id)
        
        # Retrieve the content of the selected cells using the cell IDs
        result, _ = self.tableId2Content(select_id_list, self.dealed_rows)

        # Return the selected cell contents and the corresponding cell IDs
        return result,select_id_list

    def check_arg_exists(self,arg):
        """
        # This function checks if a given argument (arg) exists in the 'cells' list.
        # It returns a tuple containing:
        # - The index (cell_id) of the argument in 'cells' if it exists.
        # - A boolean value indicating whether the argument exists (True if it exists, False otherwise).
        # 
        # Arguments:
        # - arg (any type): The value to be checked in the 'cells' list.
        #
        # Returns:
        # - A tuple (cell_id, exists), where:
        #   - 'cell_id' is the index of the argument in the 'cells' list if it exists, or None if it doesn't.
        #   - 'exists' is a boolean indicating whether the argument is present in the 'cells' list (True/False).
        """
        # Check if the provided argument 'arg' exists in the 'cells' attribute of the current object
        if arg in self.cells:
            # If 'arg' exists in the 'cells' list, find the index (position) of 'arg' in the list
            cell_id = self.cells.index(arg)
            # Return the index of the cell and True to indicate the argument was found
            return cell_id,True
        else:
            # If 'arg' is not found in the 'cells' list, return None and False to indicate the argument was not found
            return None,False

    def organize_same_row_col(self,same_row_col_list,keys):
        """
        Organizes and links cells that belong to the same row or column, based on the provided input list.

        Purpose:
            This function processes a list of items that contain cell information (either in the same row or column) 
            and organizes them by linking those that belong together. It then generates two outputs:
            - A formatted list of cell values.
            - A list of tuples representing the corresponding cells and their content.

        Args:
            same_row_col_list (list): A list where each item contains information about a cell, such as its position 
                                    and whether it belongs to the same row or column.
            keys (list): A list of keys to match with the sorted IDs of the cells.

        Returns:
            tuple:
                - `result` (list): A list of formatted strings representing the content of cells that are linked in 
                                the same row or column.
                - `tuple_result` (list): A list of tuples representing the rows and columns of linked cells, including 
                                        their content.

        Notes:
            - The function processes each item in `same_row_col_list`, checks for cells in the same row or column, 
            and links them together.
            - The function uses the `cell2Tuple` method to convert cell IDs to tuples of row, column, and content.
            - The linked cells are identified and formatted as strings for output.
            - The function generates two outputs:
                1. A list of formatted strings (`result`) containing the linked cell values.
                2. A list of tuples (`tuple_result`) containing the cell identifiers and content.

        Example:
            - If `same_row_col_list` contains cell positions that belong to the same row or column, the function 
            will merge them into a single entity, then return formatted data as `result` and `tuple_result`.
        """
        link_same_row_col = []
        # Iterate through the list and link cells in the same row or column
        for i in range(len(same_row_col_list)):
            item = copy.deepcopy(same_row_col_list[i])
            if len(item) == 0:
                continue
            # Determine the unique identifier for rows or columns based on the "SAME ROW" or "SAME COLUMN" label
            if item[1] == 'SAME ROW':
                id = item[0] // len(self.rows[0])
            else:
                id = item[0] % len(self.rows[0])

            # Compare current item with the next items to group them
            for j in range(i + 1, len(same_row_col_list)):
                n_item = same_row_col_list[j]
                if len(n_item) == 0:
                    continue
                # Calculate the identifier for the next item
                if  item[1] == 'SAME ROW':
                    n_id = n_item[0] // len(self.rows[0])
                else:
                    n_id = n_item[0] % len(self.rows[0])

                # If they belong to the same row/column and have the same ID, link them together
                if n_item[1] == item[1] and n_id == id :
                    if n_item[0] in item:
                        item.append(n_item[1])
                        item.append(n_item[2])
                    elif n_item[2] in item:
                        item.append(n_item[1])
                        item.append(n_item[0])
                    else:
                        item += [n_item[1]] + n_item
                    same_row_col_list[j] = [] # Mark the item as processed
            link_same_row_col.append(item)

        # Initialize lists to store the results
        result = []
        tuple_result = []

        # Process each key from the provided keys list
        for key_item in keys:
            for i in range(len(link_same_row_col)):
                item = link_same_row_col[i]
                id_list = sorted(list(set([k for k in item if type(k) == int]))) # Get the sorted list of cell IDs
                if key_item == id_list[0]: # Match the first item of the sorted list with the key
                    temp = []
                    tuple_item = self.cell2Tuple(id_list) # Convert the cell IDs to tuples
                    # Format the content of each cell
                    for j in range(len(id_list)):
                        cell = id_list[j]
                        row_id = cell // len(self.dealed_rows[0])
                        col_id = cell % len(self.dealed_rows[0])
                        temp.append("'{}'".format(self.dealed_rows[row_id][col_id]))
                        # tuple_item.append((row_id,col_id,"{}".format(self.dealed_rows[row_id][col_id])))
                    # Create a formatted string for the result
                    temp = '[{}]'.format( ',{},'.format(item[1]).join(temp))
                    tuple_item = '[{}]'.format( ',{},'.format(item[1]).join([str(k) for k in tuple_item]))
                    result.append(temp)
                    tuple_result.append(tuple_item)
        # Return the final result and tuple_result
        return result,tuple_result

    def cell2Tuple(self,id_list,add_id=False,add_merged_cells = False):
        """
        Converts a list of cell IDs into tuples containing the row, column, and content of each cell.

        Purpose:
            The function processes a list of cell IDs and creates tuples for each cell, where each tuple contains:
                - The row index
                - The column index
                - The content of the cell
            Optionally, it can include the cell ID and handle merged cells.

        Args:
            id_list (list): A list of cell IDs to convert into tuples.
            add_id (bool): Whether to include the cell ID in the resulting tuple (default: False).
            add_merged_cells (bool): Whether to handle merged cells and include them in the result (default: False).

        Returns:
            tuple: 
                - If `add_merged_cells` is False:
                    - `tuple_result` (list of tuples): A list of tuples, each containing (row_id, col_id, content) for each unique cell.
                - If `add_merged_cells` is True:
                    - `merged_cotent_list` (list): A list of unique content values from merged cells.
                    - `merged_id_list` (list): A list of cell IDs corresponding to the merged cells.
                    - `tuple_result` (list of tuples): A list of tuples containing (row_id, col_id, content) for all unique cells including merged ones.

        Notes:
            - The function avoids duplicate tuples in the result.
            - If `add_merged_cells` is enabled, it checks for merged cells and includes them in the result.
        """
        tuple_result = []
        merged_id_list = []
        merged_cotent_list = []

        # Iterate through each cell ID in the provided list
        for i in range(len(id_list)):
            cell_id = id_list[i]
            # Calculate the row index from the cell ID
            row_id = cell_id  // len(self.rows[0])
            # Calculate the column index from the cell ID
            col_id = cell_id % len(self.rows[0])
            # Get the content of the cell
            content = copy.deepcopy(self.dealed_rows[row_id][col_id])
            # Clean up the content
            content = str(content).replace('"', "'")

            # Create a tuple with or without the cell ID based on the `add_id` flag
            if add_id:
                temp = (cell_id,row_id,col_id,content)
            else:
                temp = (row_id, col_id, content)
            
            # Add the tuple if it's unique
            if temp not in tuple_result:
                merged_cotent_list.append(content)
                merged_id_list.append(cell_id)
                tuple_result.append(temp)

            # Handle merged cells if `add_merged_cells` is True
            if add_merged_cells:
                for rlo, rhi, clo, chi in self.merged_cells:
                    # Check if the current cell is part of a merged region
                    if row_id in range(rlo,rhi) and col_id in range(clo,chi):
                        # Add all merged cells to the result
                        for r in range(rlo,rhi):
                            for c in range(clo,chi):
                                if (r, c, content) not in tuple_result:
                                    merged_cotent_list.append(self.dealed_rows[r][c])
                                    merged_id_list.append(r*len(self.rows[0]) + c)
                                    tuple_result.append((r, c, content))
                        break
        # Return results based on whether merged cells should be included
        if add_merged_cells:
            return merged_cotent_list,merged_id_list,tuple_result

        return tuple_result






