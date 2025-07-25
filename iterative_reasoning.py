import ast
import re
from configs import PROMPT_TEMPLATE
from Generator.parse_output import get_action_list,parse_action_json, remove_quotes,LLM_json_output_parse, json_output_parser_llm_powered
from collections import OrderedDict
import copy
from tools import table2markdown, read_pre_process_tabular_meta


class GraphReasoner:
    """
    The GraphReasoner class is designed to perform iterative reasoning over a graph structure, 
    using a language model to generate and execute actions based on a given query and table data.

    Instance Variables:
    - cot_think: String for storing the chain-of-thought prompt.
    - system_instruction: String for storing system instructions for the language model.
    - dataset: The dataset name.
    - query: The query string.
    - table_caption: The caption of the table.
    - markdown_table: The table data in markdown format.
    - user_input: List of user inputs for the language model.
    - tabular_meta: Contains Summary of the table
    - use_heuristic: A variable to to confirm heurisitic modified implementation has to be run.
    - step: The current step in the iterative reasoning process.
    - max_iteration_depth: The maximum number of iterations allowed.
    - output: The final output of the reasoning process.
    - start_cells_prompt: String for storing the start cells prompt.
    - args: Arguments passed to the class.
    - search_cell_history_dict: Ordered dictionary for storing search results.
    - Intermediate_results: Ordered dictionary for storing intermediate results.
    - model: The language model used for generating prompts and responses.
    - graph_retriever: The graph retriever object used for retrieving relevant cells.
    - start_cells: String for storing the start cells.
    - connect_graph_prompt: String for storing the sub-graph prompt.
    - conn_graph_result: List for storing the sub-graph results.
    - reasoning_path: List for storing the reasoning path.
    - reasoning_path_dict: Ordered dictionary for storing the reasoning path of each step.
    - reasoning_path_prompt: String for storing the reasoning path prompt.
    """

    def __init__(self,args,model,query,table,caption,graph_retriever, table_name, dataset='hitab', is_rep=0.0):
        self.cot_think = ''
        self.system_instruction = ''

        self.dataset = dataset
        self.query = query
        self.table_caption = caption
        self.markdown_table = table
        self.user_input = []
        pre_process_smmary_of_all_tables = read_pre_process_tabular_meta(args.embed_cache_dir, args.dataset, args.model)
        self.tabular_meta = pre_process_smmary_of_all_tables[table_name]
        self.use_heuristic = args.use_heuristic

        self.step = 1
        self.max_iteration_depth = args.max_iteration_depth

        self.output=''


        self.start_cells_prompt = ''
        self.args = args


        self.search_cell_history_dict = OrderedDict() # search results
        self.Intermediate_results = OrderedDict() # Non Final results
        self.model = model
        self.graph_retriever = graph_retriever

        self.start_cells= ''
        self.connect_graph_prompt = '' # sub-graph prompt
        self.conn_graph_result = [] # sub-graph
        self.reasoning_path = []  #
        self.reasoning_path_dict = OrderedDict()  # the reasoning path of each step
        self.reasoning_path_prompt = '' # reasoning path prompt
        self.is_rep = is_rep
        return

    def initialize_prompt(self):
        cell_id_content_topk, cell_tuple_topk, conn_id_cell_dict, self.conn_graph_result = self.graph_retriever.initialize_subgraph(
            query=self.query)

        LLM_select_cells = list(cell_id_content_topk['LLM_select'].values())
        retriever_select_cells = list(cell_id_content_topk['retriever_select'].values())
        all_start_cells = LLM_select_cells + retriever_select_cells
        all_start_tuples = cell_tuple_topk['LLM_select'] + cell_tuple_topk['retriever_select']

        self.start_cells = PROMPT_TEMPLATE[self.dataset]['start_cells'].replace('{start_cells}', ','.join(
            ['{}'.format(i) for i in all_start_tuples]))

        self.connect_graph_prompt = PROMPT_TEMPLATE[self.dataset]['connect_graph'].replace('{sub_graph}', self.conn_graph_result)
        search_content = self.start_cells + '\n\n' + self.connect_graph_prompt
        print('###' * 20)
        print('retrieved cells and their connectivity graph', search_content)
        print('###' * 20)

        self.cot_think = PROMPT_TEMPLATE[self.dataset]['cot_think']

        self.system_instruction = PROMPT_TEMPLATE[self.dataset]['system_instruction']

        self.prompt = PROMPT_TEMPLATE[self.dataset]['iterative_reasoning'].replace('{question}', self.query)

        markdown_table = table2markdown(self.graph_retriever.dealed_rows)
        # markdown_table = table2Tuple(graph_retriever.dealed_rows)
        if self.table_caption:
            self.table_caption = 'Table Caption: {}\n'.format(self.table_caption)

        self.query = '**Question:** {}'.format(self.query)
        markdown_table = "**Table:**\n{}".format(markdown_table)
        self.markdown_table = self.table_caption + markdown_table

        self.retrieved_cell_id = list(cell_id_content_topk['LLM_select'].keys()) + list(
            cell_id_content_topk['retriever_select'].keys())
        self.retrieved_cell = copy.deepcopy(all_start_cells)
        self.hit_cell_neighbors_history_dict = OrderedDict()
        self.hit_cell_neighbors_history_dict[self.retrieved_cell[0]] = conn_id_cell_dict
        return

    def LLM_generate(self,  prompt, isrepeated=None, response_mime_type=None):
        if isrepeated is None:
            isrepeated = self.is_rep
            
        if 'gemini' not in self.model.model_name:
            result = self.model.generate(prompt, system_instruction=self.system_instruction, isrepeated=isrepeated,
                                    response_mime_type=response_mime_type)
        else:
            result = self.model.generate('\n'.join(prompt), system_instruction=self.system_instruction, isrepeated=isrepeated,
                                    response_mime_type=response_mime_type)
        return result
    
    def check_repeated_think_action(self, current_action_list, explanation_list,
                                    repeated_action, is_last_action_repeated, Thinking_text, Action_text):
        """
        This function checks for repeated actions in the current list of actions, based on reasoning steps. 
        If repeated actions are detected, it generates new thinking and action text prompts for the model to process. 
        It then returns the updated action list, explanations, and flags for repeated actions.

        Purpose:
        - The function iterates through the provided `current_action_list` and identifies if any action has been repeated 
        based on previous reasoning steps stored in `self.reasoning_path_dict`.
        - If a repeated action is detected, the function generates new thinking and action prompts for the model to generate 
        updated responses (e.g., `Thinking_text` and `Action_text`).
        - It also updates flags such as `is_last_action_repeated` and appends repeated actions to the `repeated_action` list.

        Parameters:
        - current_action_list: A list of actions to be checked for repetition.
        - explanation_list: A list of explanations corresponding to each action.
        - repeated_action: A list of previously detected repeated actions.
        - is_last_action_repeated: A flag indicating if the last action was repeated.
        - Thinking_text: Text containing the reasoning or thought process generated for the current step.
        - Action_text: Text containing the action steps generated for the current step.

        Outputs:
        - result_action_list: The updated list of actions after checking for repeated actions and generating new ones.
        - explanation_list: The updated list of explanations corresponding to the actions.
        - repeated_action: The updated list of detected repeated actions.
        - is_last_action_repeated: The updated flag indicating if the last action was repeated.
        - Thinking_text: The newly generated thinking text for the current step, if needed.
        - Action_text: The newly generated action text for the current step, if needed.

        Explanation:
        - The function first converts the `current_action_list` into a list of unique actions by checking the combination of 
        `action_type` and `argument` for each action.
        - It checks whether any action has already been performed in the reasoning steps (from `self.reasoning_path_dict`). 
        If a repeated action is found and its reasoning step is more than 1 step away from the current step, it is added to 
        the `repeated_action` list.
        - If the last action was repeated or if the `current_action_list` is empty, the function generates new prompts for 
        thinking and actions, calls the model to generate updated `Thinking_text` and `Action_text`, and updates the 
        `result_action_list` and `explanation_list` accordingly.
        - The `is_last_action_repeated` flag is reset to `False` after the processing.

        """
        action_list = []
        result_action_list = []
        # convert format and store
        for j in range(len(current_action_list)):
            action = current_action_list[j]
            action_type, argument = parse_action_json(action)
            argument = '|'.join([str(i) for i in argument])
            if [action_type, argument] not in action_list:
                action_list.append([action_type, argument])
                result_action_list.append(action)
        for i in range(len(action_list)):
            action = action_list[i]
            for b_step, act in reversed(self.reasoning_path_dict.items()):
                if action in act:
                    if abs(self.step - b_step) > 1:
                        # repeated_action.append([step,temp_action_list[i],b_step])
                        repeated_action.append(
                            [self.step, f"{current_action_list[i]['function_name']}{str(current_action_list[i]['parameters'])}",
                             b_step])
                    else:
                        is_last_action_repeated = True
        if is_last_action_repeated or len(current_action_list) == 0:
            if is_last_action_repeated:
                prompt = copy.deepcopy(self.user_input[:-2]) + [
                    PROMPT_TEMPLATE[self.dataset]['think_prompt'].replace('{step}', str(self.step))]

                Thinking_text = self.LLM_generate(prompt, isrepeated=0.7)

                # Thinking_text = Thinking['text']
                Action_prompt = PROMPT_TEMPLATE[self.dataset]['action_prompt'].replace('{step}', str(self.step))

                prompt = prompt[:-1] + ['\n' + Thinking_text.strip().strip('\n'), '\n' + Action_prompt]
            else:
                print(str(self.step) + str(current_action_list) + 'Action is empty')
                prompt = copy.deepcopy(self.user_input)

            Action_text = self.LLM_generate(prompt, isrepeated=0.7)

            # Action_text = Action['text']


            result_action_list, explanation_list = get_action_list(Action_text, self.model)
            is_last_action_repeated = False

            return result_action_list, explanation_list, repeated_action, is_last_action_repeated , Thinking_text, Action_text

        return result_action_list, explanation_list, repeated_action, is_last_action_repeated, Thinking_text, Action_text

    def Thought(self):
        Thinking_prompt = PROMPT_TEMPLATE[self.dataset]['think_prompt'].replace('{step}', str(self.step))
        self.user_input = [
            self.prompt,
            self.cot_think,
            self.markdown_table  if not self.use_heuristic else self.tabular_meta,
            '\n' + self.query + '\n',
            self.start_cells + '\n' if self.step == 1 else self.reasoning_path_prompt + '\n',
            self.connect_graph_prompt,
        ]
        if self.step > 1:
            self.user_input += ['\n' + self.Intermediate_results[self.step - 1]['think'],
                                self.Intermediate_results[self.step - 1]['action'],
                                self.Intermediate_results[self.step - 1]['interaction_prompt']]
        self.user_input += ['\n' + Thinking_prompt]
        # self.user_input.append('\n'+Thinking_prompt)
        # iterative_process.append('\n'.join(self.user_input))

        if self.args.debug:
            print('Thought prompt：','\n'.join(self.user_input))

        # dialogs = [{"role": "user", "content": '\n'.join(self.user_input)}]

        Thinking = self.LLM_generate(self.user_input)
        return Thinking

    def Action(self,Thinking_text):
        Action_prompt = PROMPT_TEMPLATE[self.dataset]['action_prompt'].replace('{step}', str(self.step))

        self.user_input = [
            self.prompt,
            self.markdown_table  if not self.use_heuristic else self.tabular_meta,
            '\n' + self.query + '\n',
            self.start_cells + '\n' if self.step == 1 else self.reasoning_path_prompt + '\n',
            self.connect_graph_prompt,
        ]
        if self.step > 1:
            self.user_input += ['\n' + self.Intermediate_results[self.step - 1]['think'],
                                self.Intermediate_results[self.step - 1]['action'],
                                self.Intermediate_results[self.step - 1]['interaction_prompt']]
        self.user_input += ['\n' + Thinking_text.strip().strip('\n') + '\n', Action_prompt]

        print('---' * 30)
        if self.args.debug:
            print('\n'.join(self.user_input))
            print('---' * 30)

        Action = self.LLM_generate(self.user_input,
                                   response_mime_type="application/json" if 'gemini' in self.model.model_name else {
                                       "type": "json_schema"})
        return Action

    def get_cell_id(self,argument):
        """
        Retrieve the cell ID for a given argument, which could be an alias or a neighboring cell of a previously 
        retrieved cell. This function searches through different histories to find a matching cell and return its ID.

        Purpose:
        - The function checks whether the given argument (e.g., a cell identifier) has already been retrieved and stored.
        - If the argument is found in one of the histories (retrieved cells, search history, or neighboring cells), 
        it retrieves the corresponding cell ID.
        - If the argument is not found in any history, it returns a flag indicating that the cell ID does not exist.

        Parameters:
        - argument: The identifier or alias of a cell whose ID needs to be retrieved.

        Outputs:
        - cell_id_exist (bool): A flag indicating whether the cell ID was found (True) or not (False).
        - cell_id (int or None): The corresponding cell ID if found, otherwise None.

        Explanation:
        - First, the function checks if the argument is already in the `retrieved_cell` list, indicating it has been previously retrieved.
        - If not found, it looks into the `search_cell_history_dict` to check if the argument is an alias of an already retrieved cell.
        - It then checks the `hit_cell_neighbors_history_dict` to see if the argument is a neighboring cell of any previously retrieved cell.
        - If the argument is found in any of these histories, the function retrieves and returns the corresponding cell ID.
        - If the argument is not found, the function sets the cell ID as None and marks `cell_id_exist` as False.

        """
        if argument in self.retrieved_cell:  # if this cell is an already retrieved cell
            cell_index = self.retrieved_cell.index(argument)
            cell_id = self.retrieved_cell_id[cell_index]
        else:
            is_arg_in_search_his = False
            if len(self.search_cell_history_dict) > 0:  # If this cell is an alias of one of the already retrieved cells
                for hit_cell_id, his in self.search_cell_history_dict.items():
                    if is_arg_in_search_his:
                        is_arg_in_search_his = False
                        break
                    for k, v in his.items():
                        if k == argument:
                            is_arg_in_search_his = True
                            cell_id = hit_cell_id
                            break
            if len(self.hit_cell_neighbors_history_dict) > 0 and not is_arg_in_search_his:  #if this cell is a neighboring cell of one of the already retrieved cells
                is_hit = False
                for _, neighbors in self.hit_cell_neighbors_history_dict.items():
                    if is_hit:
                        is_hit = False
                        break
                    for k, v in neighbors.items():
                        if v == argument:
                            cell_id = k
                            is_hit = True
                            break
    
        # Check if 'cell_id' is defined; 
        # set 'cell_id_exist' to False flag and assign 'None' if not defined
        try:
            cell_id
        except NameError:
            cell_id_exist = False
            cell_id = None
        else:
            cell_id_exist = True
        return cell_id_exist,cell_id


    def Answer(self, Thinking_text, answer_explan='', last_interact_step=''):
        final_answer_prompt = PROMPT_TEMPLATE[self.dataset]['LLM_final_answer']
        cot_answer = PROMPT_TEMPLATE[self.dataset]['cot_answer']

        self.user_input = [
            cot_answer,
            final_answer_prompt + '\n',
            self.markdown_table  if not self.use_heuristic else self.tabular_meta,
            '\n' + self.query + '\n',
            self.start_cells + '\n',
            self.connect_graph_prompt,
        ]
        if self.step > 1:
            self.user_input += ['\n' + self.Intermediate_results[self.step - 1]['think'],
                           self.Intermediate_results[self.step - 1]['action'],
                           self.Intermediate_results[self.step - 1]['interaction_prompt']]
        self.user_input += ['\n' + Thinking_text.strip().strip('\n') + '\n' + answer_explan, last_interact_step,
                       PROMPT_TEMPLATE[self.dataset]['LLM_final_answer_format']]
        if self.args.debug:
            print('\n'.join(self.user_input))
        output = self.LLM_generate(self.user_input)

        print('LLM\'s response', output)


        # answer = get_answer(output['text'])
        try:
            parsed_output = LLM_json_output_parse(output)
        except Exception as e:
            parsed_output = LLM_json_output_parse(json_output_parser_llm_powered(output, self.model))

        answer = parsed_output['answer']
        # answer = answer_calculator(model, calculator, query, answer, dataset)

        return answer

    def filter_actions(self,current_action_list, explanation_list):
        """
        Filters the provided lists of actions and explanations by excluding actions 
        whose 'function_name' contains the word 'answer'. If more than one action is 
        provided, only the actions that do not contain 'answer' in their 'function_name' 
        are returned along with their corresponding explanations.

        Purpose:
        - The function iterates through the `current_action_list` and filters out actions 
        that have the word 'answer' in their `function_name`. It also removes the corresponding 
        explanations in `explanation_list` for those actions.
        - If there is only one action, no filtering is applied, and the original lists are returned.

        Parameters:
        - current_action_list: A list of dictionaries representing actions, where each action has a 'function_name'.
        - explanation_list: A list of explanations corresponding to each action.

        Outputs:
        - filtered_actions (list): A list of actions after filtering out those with 'answer' in their 'function_name'.
        - filtered_explanations (list): A list of explanations corresponding to the filtered actions.

        Explanation:
        - The function checks if the length of `current_action_list` is greater than 1. If so, it iterates over each action and 
        its corresponding explanation. Actions with 'answer' in their `function_name` are excluded from the result.
        - If the length of `current_action_list` is 1 or less, the function simply returns the original lists without modification.
        - We have an intuition that last action will always be answer, so no filteration will be required when the list of actions.
        has size 1, as it will be the answer action.
        """
        if len(current_action_list) > 1:
            temp = []
            temp_ = []
            for i in range(len(current_action_list)):

                if 'answer' not in current_action_list[i]['function_name'].lower():
                    temp.append(current_action_list[i])
                    temp_.append(explanation_list[i])
            return temp,temp_
        return current_action_list, explanation_list

    def tuple2cell(self,argument):
        """
        Convert a given argument (either a string representing a tuple or an actual tuple) 
        into a cell ID in the graph, or retrieve the cell ID based on the input argument.

        Purpose:
        - The function takes an argument which could either be a string representing a tuple (e.g., '(row, col)') 
        or a tuple itself (e.g., (row, col)). 
        - It parses the argument to determine the corresponding cell ID in a graph, based on row and column numbers.
        - If the argument is neither a string nor a tuple, it uses another method to get the cell ID.

        Parameters:
        - argument: A string (representing a tuple) or a tuple (row, col) representing the location of the cell.

        Outputs:
        - nei_cell_id_exist (bool): A flag indicating whether the cell ID exists.
        - nei_cell_id (int): The computed cell ID based on the row and column, or retrieved by another method if the argument is invalid.

        Explanation:
        - If the argument is a string starting with '(' and ending with ')', it is treated as a string representation of a tuple.
        - The string is then converted into an actual tuple using `ast.literal_eval` if necessary.
        - The row and column values are extracted from the tuple, and a cell ID is computed using these values.
        - If the argument is already a tuple, it is processed directly.
        - If the argument is neither a string nor a tuple, the method calls `get_cell_id` to retrieve the cell ID.

        """
        if (type(argument) == str and argument.startswith('(') and argument.endswith(')')) or type(
                argument) == tuple:
            nei_cell_tuple = ast.literal_eval(argument) if type(argument) == str else argument
            nei_cell_id_exist = True

            col_num = int(nei_cell_tuple[1])
            row_num = int(nei_cell_tuple[0])

            nei_cell_id = len(self.graph_retriever.dealed_rows[0]) * row_num + col_num
        elif (type(argument) == str and argument.startswith('{') and argument.endswith('}')) or type(
                argument) == dict:
            nei_cell_tuple = ast.literal_eval(argument) if type(argument) == str else argument
            column_keys = {'col', 'COL', 'Col', 'COLOUMN', 'coloumn', 
                           'Coloumn', 'column', 'Column', 'COLUMN', 'ColoumnIndex', 
                           'colIndex', 'coloumnIndex', 'columnIndex', 'ColumnIndex', 'Coloumn_Index', 
                           'col_index', 'coloumn_index', 'column_index', 'Column_Index'}
            row_keys = {'row', 'ROW', 'Row', 'ROWS', 'rows', 
                        'Rows', 'RowIndex', 'rowIndex', 'Row_Index', 'row_index', 
                        'Row_Index'}

            col_index_value = next((nei_cell_tuple[key] for key in column_keys if key in argument), None)
            row_index_value = next((nei_cell_tuple[key] for key in row_keys if key in nei_cell_tuple), None)
            if col_index_value is None or row_index_value is None:
                nei_cell_id_exist, nei_cell_id = self.get_cell_id(argument)
            else:
                nei_cell_id_exist = True
                col_num = int(col_index_value)
                row_num = int(row_index_value)
                nei_cell_id = len(self.graph_retriever.dealed_rows[0]) * row_num + col_num
        else:
            nei_cell_id_exist, nei_cell_id = self.get_cell_id(argument)

        return nei_cell_id_exist, nei_cell_id

    def iterative_reasoning(self):
        """
        This function simulates an iterative reasoning process where the system thinks, takes actions, 
        and retrieves information step by step. It processes a series of actions (such as Answer, 
        SearchNode, GetAllNeighbours, GetSharedNeighbours) and builds upon each action to 
        make more informed decisions.

        The actions involved are:

        1. **Answer**:
        - Purpose: Provides an answer based on the reasoning process.
        - Example: If the reasoning process leads to a conclusion, it returns the answer with an explanation.
        - Action Type: 'Answer'
        - Example Explanation: "Function: Answer(function_name=retrieve_answer, parameters=[question], 
                                Explanation: Answer to the question is '42'"

        2. **SearchNode**:
        - Purpose: Searches for a specific node or cell based on a given argument.
        - Example: Given an argument (e.g., a query string), the function searches a graph for the matching node.
        - Action Type: 'SearchNode'
        - Example Explanation: "Function: SearchNode(function_name=search_node, parameters=['find_node'], 
                                Explanation: Node 'find_node' found at position (x, y)."

        3. **GetAllNeighbours**:
        - Purpose: Retrieves all neighboring cells for a given node.
        - Example: If the argument is a cell ID, this action returns all neighboring cells of that ID.
        - Action Type: 'GetAllNeighbours'
        - Example Explanation: "Function: GetAllNeighbours(function_name=get_all_neighbours, parameters=[cell_id], 
                                Explanation: Neighbors of cell are [(x1, y1), (x2, y2), (x3, y3)]."

        4. **GetSharedNeighbours**:
        - Purpose: Retrieves the common neighboring cells shared between two nodes.
        - Example: Given two cell IDs, this action finds the shared neighbors between both.
        - Action Type: 'GetSharedNeighbours'
        - Example Explanation: "Function: GetSharedNeighbours(function_name=get_shared_neighbours, parameters=[cell1, cell2], 
                                Explanation: Shared neighbors are [(x1, y1), (x2, y2)]."

        In this function, each action is checked for repetition, and if an action is repeated, a thinking process is triggered 
        to avoid redundant steps. Based on the type of action, the system performs tasks like searching for nodes, 
        retrieving neighbors, or providing an answer.

        """

        # Initialize the model prompt for reasoning
        self.initialize_prompt()

        # Track repeated actions and whether the last action was repeated
        repeated_action = []
        is_last_action_repeated = False

        # Start an infinite loop to simulate iterative reasoning
        while True:
            
            # Get the model's current thinking process (output)
            Thinking_text = self.Thought()

            # No need for the following line. 
            # Thinking_text = Thinking['text']

            print('the model\'s thinking steps are {}'.format(Thinking_text))

            # Get the action(s) generated by the model based on its thinking
            Action_text = self.Action(Thinking_text)

            # No need for the following line. 
            # Action_text = Action['text'].replace(f'Action Step {self.step}:', '').replace(f'Action step {self.step}:', '')

            print('the model\'s action steps are: {}'.format(Action_text))

            # Reset reasoning path for the current step
            self.reasoning_path_dict[self.step] = []
            interaction_result = []

            # xtract the list of actions and corresponding explanations from the model's action output
            current_action_list, explanation_list = get_action_list(Action_text, self.model)

            # Check and handle repeated thinking and actions
            current_action_list, explanation_list, repeated_action, \
            is_last_action_repeated, Thinking_text, Action_text = self.check_repeated_think_action(current_action_list,
                                                                                                    explanation_list,
                                                                                                    repeated_action,
                                                                                                    is_last_action_repeated,
                                                                                                    Thinking_text,
                                                                                                    Action_text)

            # Filter out any irrelevant or unwanted actions
            current_action_list, explanation_list = self.filter_actions(current_action_list, explanation_list)

            current_explanation_list = []
            if current_action_list:
                one_step_path = []

                # Process each action in the current list
                for t_action in range(len(current_action_list)):
                    tmp_action = current_action_list[t_action]
                    try:
                        action_type, argument = parse_action_json(tmp_action)
                        tmp_action = f"{tmp_action['function_name']}({', '.join([str(i) for i in tmp_action['parameters']])})"
                    except Exception as e:
                        print(f'There is something wrong with the generated target actions {tmp_action}.')
                        raise Exception(f"Action parsing error {e.__str__()}")
                    
                    # Handle actions of type 'Answer'
                    if action_type == 'Answer' or 'Answer' in action_type:
                        answer_explan_pattern = r'Explanation:(.*)'
                        match = re.search(answer_explan_pattern, explanation_list[t_action])
                        answer_explan = match.group(1).strip('\n') if match else explanation_list[t_action].strip('\n')
                        answer = self.Answer(
                            Thinking_text=Thinking_text, answer_explan=answer_explan)
                        # return answer, step, prompt_length, iterative_process

                        # Return the answer if found
                        answer = ', '.join([str(i) for i in answer])
                        return answer, self.step

                    # Handle actions of type 'SearchNode' (searching for relevant data)
                    elif action_type == 'SearchNode':
                        # one_step_path.append(argument)
                        argument = str(argument[0])
                        search_cell, search_cell_id, search_cell_tuple = self.graph_retriever.search_cell(query=argument,
                                                                                                     topk=1)
                        if len(search_cell_id) > 0:
                            self.search_cell_history_dict[search_cell_id[0]] = {argument: search_cell[0]}
                        
                        # No Need for the following line
                        # interaction_result.append([tmp_action, search_cell])

                        # Append the search results to the interaction
                        interaction_result.append([tmp_action, search_cell_tuple])
                        self.reasoning_path_dict[self.step].append([action_type, '|'.join([str(i) for i in [argument]])])

                        one_step_path.append(tmp_action)
                        current_explanation_list.append(explanation_list[t_action])
                        # Update the retrieved cell history if applicable
                        if len(search_cell) > 0 and search_cell[0] not in self.retrieved_cell:
                            self.retrieved_cell.append(search_cell[0])
                            self.retrieved_cell_id.append(search_cell_id[0])
                            hit_cell_same_row_col = self.graph_retriever.getSameRow_ColCells(match_cell_id=self.retrieved_cell_id)
                            _, _, self.conn_graph_result = self.graph_retriever.hit_cell_connect_graph(hit_cell_same_row_col)

                    # Handle actions of type 'GetAllNeighbours' (retrieving all neighboring cells)
                    elif action_type == 'GetAllNeighbours':
                        argument = argument[0]

                        # Retrieve the neighboring cells for the given argument
                        nei_cell_id_exist, nei_cell_id = self.tuple2cell(argument)

                        if not nei_cell_id_exist:
                            nei_cell_id, nei_cell_id_exist = self.graph_retriever.check_arg_exists(argument)
                        if not nei_cell_id_exist:
                            print('GetAllNeighbours could not find the cell', argument)
                            raise Exception(f'GetAllNeighbours could not find the cell {argument}')
                        else:
                            # Get neighbors for the identified cell
                            cell_topk, nei_cells, hit_cell_neighbors_content_id = self.graph_retriever.get_neighbors(
                                add_id_list=[nei_cell_id],
                                get_same_row=False,
                                get_all_nei=True)
                            self.hit_cell_neighbors_history_dict[nei_cell_id] = hit_cell_neighbors_content_id

                            interaction_result.append([tmp_action, '\n' + nei_cells])
                            self.reasoning_path_dict[self.step].append([action_type, '|'.join([str(i) for i in [argument]])])
                            one_step_path.append(tmp_action)
                            current_explanation_list.append(explanation_list[t_action])

                            # Update the retrieved cell history if applicable
                            if len(cell_topk) > 0 and cell_topk[0] not in self.retrieved_cell:
                                self.retrieved_cell.append(cell_topk[0])
                                self.retrieved_cell_id.append(nei_cell_id)
                                hit_cell_same_row_col = self.graph_retriever.getSameRow_ColCells(
                                    match_cell_id=self.retrieved_cell_id)
                                _, _, self.conn_graph_result = self.graph_retriever.hit_cell_connect_graph(hit_cell_same_row_col)
                            del nei_cell_id

                    # Handle actions of type 'GetSharedNeighbours' (retrieving shared neighbors between two cells)
                    elif action_type == 'GetSharedNeighbours':
                        try:
                            cell1, cell2 = argument if len(argument) == 2 else argument[:2]
                            cell1 = remove_quotes(cell1) if type(cell1) == str else cell1
                            cell2 = remove_quotes(cell2) if type(cell2) == str else cell2
                        except Exception as e:
                            print('GetSharedNeighbours Parsing error', argument)
                            raise Exception(f'GetSharedNeighbours Parsing error {argument}')
                        else:
                            # Retrieve shared neighbors between the two cells
                            cell1_exists, cell1_id = self.tuple2cell(cell1)
                            cell2_exists, cell2_id = self.tuple2cell(cell2)

                            if not cell1_exists:
                                cell1_id, cell1_exists = self.graph_retriever.check_arg_exists(cell1)
                            if not cell2_exists:
                                cell2_id, cell2_exists = self.graph_retriever.check_arg_exists(cell2)
                            if not cell1_exists or not cell2_exists:

                                print('GetSharedNeighbours Cell not found', argument)
                                raise Exception(f'GetSharedNeighbours Could not find the cell {argument}')
                            else:
                                # Get shared neighbors between the cells
                                hit_cell_same_row_col = self.graph_retriever.getSameRow_ColCells(
                                    match_cell_id=[cell1_id, cell2_id])
                                _, connect_id_cell_dict, cell_shared_neighbors = self.graph_retriever.hit_cell_connect_graph(
                                    hit_cell_same_row_col, get_shared_nei=True)

                                interaction_result.append([tmp_action, '\n' + cell_shared_neighbors])
                                self.reasoning_path_dict[self.step].append([action_type, '|'.join([str(i) for i in argument])])
                                one_step_path.append(tmp_action)
                                current_explanation_list.append(explanation_list[t_action])

                                self.hit_cell_neighbors_history_dict[cell1_id] = connect_id_cell_dict
                
                # Add reasoning path for the current step
                self.reasoning_path.append(str(one_step_path))
            
            # Format and display the interaction result for this iteration
            interaction_result = '\n'.join(
                ['{}. The result of {} is: {}'.format(i + 1, interaction_result[i][0], interaction_result[i][1]) for i
                 in
                 range(len(interaction_result))])

            print('****' * 20)
            print('The query result is：', interaction_result)
            print('****' * 20)

            # Check for repeated actions and update the prompt accordingly
            if len(repeated_action) > 0 and not is_last_action_repeated:
                repeated_action_prompt = PROMPT_TEMPLATE[self.dataset]['repeated_action_attention'].replace('{step}',
                                                                                                       str(self.step)) \
                    .replace('{action}', ','.join([str(i[1]) for i in repeated_action])) \
                    .replace('{last_step}', ','.join(list(set([str(i[2]) for i in repeated_action]))))
                Interaction_prompt = """Observation Step {}:\n{}\n\n{}""".format(str(self.step), interaction_result,
                                                                                 repeated_action_prompt)
                repeated_action = []
            else:
                Interaction_prompt = """Observation Step {}:\n{}""".format(str(self.step), interaction_result)

            # Store intermediate results for this step
            self.Intermediate_results[self.step] = {
                'think': Thinking_text.strip().strip('\n').replace('\n', ''),
                'action': f"Action Step {self.step}:\n" + '\n'.join(
                    ['{}. {}'.format(i + 1, current_explanation_list[i]) for i in range(len(current_explanation_list))]),
                'interaction_prompt': Interaction_prompt,

            }

            # Format and store reasoning path
            reasoning_steps = '\n'.join(
                ['Step {}:'.format(i + 1) + str(self.reasoning_path[i]) for i in range(len(self.reasoning_path))])
            self.reasoning_path_prompt = PROMPT_TEMPLATE[self.dataset]['reasoning_path'].replace('{start_cells}', self.start_cells) \
                .replace('{reasoning_steps}', reasoning_steps)

            self.connect_graph_prompt = PROMPT_TEMPLATE[self.dataset]['connect_graph'].replace('{sub_graph}', self.conn_graph_result)

            # Format and store connect graph prompt
            if self.step > self.max_iteration_depth:
                print("The number of iterations has exceeded {} times; this inference must provide an answer".format(self.max_iteration_depth))

                answer = self.Answer(Thinking_text=Thinking_text,last_interact_step=f"{self.Intermediate_results[self.step]['action']}\n{self.Intermediate_results[self.step]['interaction_prompt']}\n")
                answer = ', '.join([str(i) for i in answer])

                return answer, self.step
            # Increment the step for the next iteration
            self.step += 1
