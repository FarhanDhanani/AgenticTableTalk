import os
import logging
import pickle
import faiss
import numpy as np
import torch
from copy import deepcopy

import sentence_transformers
from langchain_ollama import OllamaEmbeddings

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# NODE_TEXT_KEYS = {'maple': {'paper': ['title'], 'author': ['name'], 'venue': ['name']},
#                   'amazon': {'item': ['title'], 'brand': ['name']},
#                   'biomedical': {'Anatomy': ['name'], 'Biological_Process':['name'], 'Cellular_Component':['name'], 'Compound':['name'], 'Disease':['name'], 'Gene':['name'], 'Molecular_Function':['name'], 'Pathway':['name'], 'Pharmacologic_Class':['name'], 'Side_Effect':['name'], 'Symptom':['name']},
#                   'legal': {'opinion': ['plain_text'], 'opinion_cluster': ['syllabus'], 'docket': ['pacer_case_id', 'case_name'], 'court': ['full_name']},
#                   'goodreads': {'book': ['title'], 'author': ['name'], 'publisher': ['name'], 'series': ['title']},
#                   'dblp': {'paper': ['title'], 'author': ['name', 'organization'], 'venue': ['name']}
#                   }
NODE_TEXT_KEYS = {
    'table':{'table':['value'],'row':['value'],'header_cell':['value'],'data_cell':['value']}
}

class Retriever:
    """
    The Retriever class is designed to load and manage embedding models, process graph data, and perform 
    dense retrieval operations. It supports both sentence-transformers and OllamaEmbeddings models.

    Instance Variables:
    - use_gpu: Boolean indicating whether to use GPU for computations.
    - device: String specifying the device to use ('cuda' for GPU, 'cpu' otherwise).
    - model_name: Path to the embedding model.
    - model: The embedding model instance.
    - cache: Boolean indicating whether to use caching.
    - cache_dir: Directory path for storing cache files.
    - graph: The graph data structure representing the table.
    - dataset: The dataset name.
    - table_name: The name of the table.
    - doc_lookup: List of document IDs.
    - doc_type: List of document types.
    - index: FAISS index for similarity search.
    """

    def __init__(self, args, cache=True, graph=None):
        logger.info("Initializing retriever")

        self.use_gpu = args.gpu
        self.device  = 'cuda' if self.use_gpu else 'cpu'
        self.model_name = args.embedder_path
        self.load_ollama_embedder(args.embedder_path)
        if cache and args.embed_cache_dir is not None:
            self.cache = cache
            self.cache_dir = args.embed_cache_dir
        # No need for the following lines in the current implementation
        # self.node_text_keys = args.node_text_keys
        # self.graph = graph
        # self.reset()
        return

    def load_embedder(self, embedder_path):
        try:
            self.model = sentence_transformers.SentenceTransformer(embedder_path, device=self.device, use_auth_token="your hf token")
            logger.info("Embedder Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
        return

    def load_ollama_embedder(self, embedder_path):
        try:
            self.model = OllamaEmbeddings(model=embedder_path)
            logger.info("Embedder Model loaded successfully from Ollama")
        except Exception as e:
            logger.error(f"Error loading model from Ollama: {e}")
            raise e
        return
    
    def load_graph(self,graph,dataset,table_name):
        """
        Load a graph, dataset, and table name, and reset the system for the new data.

        Purpose:
        - This function is responsible for loading a new graph along with its associated dataset and table name into the system.
        - It sets the internal attributes for the graph, dataset, and table, then triggers a reset to initialize or update the model with the new data.

        Parameters:
        - graph (dict): The graph data, which typically contains nodes, edges, and other related information.
        - dataset (str): The name of the dataset being processed.
        - table_name (str): The name of the table corresponding to the dataset and graph.

        Outputs:
        - None. The function does not return any value but updates the internal state of the class with the provided graph, dataset, and table name, and resets the model.

        Notes:
        - After loading the graph and setting the dataset and table name, the `reset()` method is called to reprocess the graph, cache embeddings, and reinitialize the index if needed.
        """

        self.graph = graph
        self.dataset = dataset
        self.table_name = table_name

        self.reset()
        return
    
    def reset(self):
        """
        Resets the model by either loading cached embeddings or generating new ones for the given dataset.

        Purpose:
        - This function is responsible for processing the graph data, embedding the documents, 
        and either loading previously cached embeddings or computing and saving new ones.
        - The goal is to avoid redundant computations by reusing embeddings stored in the cache.
        - The function also initializes or updates the index for efficient search or retrieval.

        Steps:
        1. **Process the graph:** The function first processes the graph data by calling `process_graph()`,
        which returns documents, ids, and meta information.
        2. **Check cache:** It checks if a valid cached embeddings file exists for the current model, dataset, and table.
        3. **Load from cache or compute embeddings:** 
            - If the cache is available, it loads the embeddings, `doc_lookup` (IDs), and `doc_type` (meta information) from the cache.
            - If no cache is available, it generates new embeddings by calling `embed_docs()`, stores them along with the IDs and meta information, and saves them to a cache file.
        4. **Index initialization:** Finally, the function initializes or updates the index using the embeddings.

        Parameters: 
        - None. The function relies on class attributes such as `model_name`, `cache`, `cache_dir`, etc.

        Outputs:
        - None. The function does not return any value. It modifies the state of the class by updating the embeddings,
        document lookup, document type, and index.

        Notes:
        - The cache file name is dynamically generated based on the model name, dataset, and table name.
        - The `assert` statements ensure that the loaded cache's document lookup and type match the current ones.

        """
        docs, ids, meta_type = self.process_graph()
        save_model_name = self.model_name.split('/')[-1]

        if self.cache and os.path.isfile(os.path.join(self.cache_dir, f'cache-{save_model_name}-{self.dataset}-{self.table_name}.pkl')):
            embeds, self.doc_lookup, self.doc_type = pickle.load(open(os.path.join(self.cache_dir, f'cache-{save_model_name}-{self.dataset}-{self.table_name}.pkl'), 'rb'))
            assert self.doc_lookup == ids
            assert self.doc_type == meta_type
        else:
            embeds = self.embed_docs(docs)
            self.doc_lookup = ids
            self.doc_type = meta_type
            pickle.dump([embeds, ids, meta_type], open(os.path.join(self.cache_dir, f'cache-{save_model_name}-{self.dataset}-{self.table_name}.pkl'), 'wb'))

        self.init_index_and_add(embeds)
        return

    def process_graph(self):
        """
        Process the graph data structure and extract relevant information to generate document-like representations.

        Purpose:
        - This function processes a graph-like data structure stored in `self.graph`, 
        iterating over its rows to generate three outputs:
            1. A list of documents (`docs`), each representing cellular data extracted from the graph rows.
            2. A list of unique IDs (`ids`), one for each cells, used to identify the corresponding document.
            3. A list of metadata types (`meta_type`), assigning a label ("Cell" in this case) to each cells as is.

        Outputs:
        - docs: A list of strings, where each string corresponds to a cell in the graph data.
        - ids: A list of integers, representing unique indices for each cell in the graph.
        - meta_type: A list of strings, with the constant value "Cell" indicating the type of each cell.
        
        Notes:
        - In the case of the current implementation, the graph contains cellular level data opposed to row-level data.
        So, in the current implementation, the function processes the cells of the tables in the name rows.
        - The function includes a commented-out section that appears to handle a more detailed graph 
        structure, extracting node-level embeddings based on text features. However, the current 
        implementation processes the graph rows as-is.
        """
        docs = []
        ids = []
        meta_type = []

        # Originally based on graph embeddings
        # for node_type_key in self.graph.keys():
        #     node_type = node_type_key.split('_nodes')[0]
        #     logger.info(f'loading text for {node_type}')
        #     for nid in tqdm(self.graph[node_type_key]):
        #         docs.append(str(self.graph[node_type_key][nid]['features'][self.node_text_keys[node_type][0]]))
        #         ids.append(nid)
        #         meta_type.append(node_type)
        index = 0
        for cell in self.graph:
            # for cell in graph:
            ids.append(index)
            docs.append(cell)
            meta_type.append('Cell')
            index += 1
        return docs, ids, meta_type
    
    def embed_query(self, query):
        """
        Embed a single query into a numerical vector space using the specified model.

        Purpose:
        - This function takes a single textual query and generates its corresponding embedding using the model specified in `self.model`.
        - It supports different embedding models, such as `OllamaEmbeddings` and sentence-transformers, for encoding the query into a vector.

        Parameters:
        - query (str): A textual query that needs to be embedded into a numerical vector.

        Outputs:
        - query_embed (numpy.ndarray or similar): A numerical embedding of the query, represented as a vector.
        
        Notes:
        - The function checks the type of `self.model` and calls the appropriate method to embed the query:
            - If the model is of type `OllamaEmbeddings`, it uses `embed_query` for embedding.
            - Otherwise, it uses `encode` from sentence-transformers or similar models to generate the embedding.
        - If any error occurs during the embedding process, it is logged, and the exception is re-raised.

        """

        try:
            if isinstance(self.model, OllamaEmbeddings):
                query_embed = self.model.embed_query(query)
                query_embed = np.array(query_embed)
            else:
                # if the model is type of sentence-transformers
                query_embed = self.model.encode(query, show_progress_bar=False)
            return query_embed
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            raise e 
        return      

    def embed_docs(self, docs):
        """
        Embed a list of documents into a numerical vector space using the specified model.

        Purpose:
        - This function takes a list of documents (`docs`) and generates their corresponding 
        embeddings using the `self.model` instance. It supports multiple embedding models, 
        including `OllamaEmbeddings` and sentence-transformers.
        - Handles model-specific embedding methods and potential compatibility issues, 
        ensuring smooth execution.

        Parameters:
        - docs (list of str): A list of textual documents to be embedded.

        Outputs:
        - embeds (numpy.ndarray): A 2D array where each row corresponds to the embedding of a document.

        Notes:
        - The function includes commented-out code for multi-process embedding with sentence-transformers,
        which may improve performance for large datasets.
        - A known issue with `numpy` version 2.2.1 caused compatibility problems with sentence-transformers. 
        Downgrading `numpy` to version 1.26.4 resolved the issue.
        - Error handling is implemented to log and raise exceptions if the embedding process fails.
        """
        # Multi-process embedding with sentence-transformers only
        # pool = self.model.start_multi_process_pool()
        # embeds = self.model.encode_multi_process(docs, pool)

        # I was experiencing some issues in excecuting the following flow with sentence transformers 
        # and numpy 2.2.1 version apprently by downgrading numpy to 1.26.4 I was able to successfully 
        # resolve the issue.
        try:
            if isinstance(self.model, OllamaEmbeddings):
                embeds = self.model.embed_documents(docs)
                embeds = np.array(embeds)  # Convert to numpy array
            else:
                # Incase of Sentene-Transformers are used.
                embeds = self.model.encode(docs, 
                                        device=self.device, 
                                        show_progress_bar=True,
                                        batch_size=1,)
            print("Embedded sucessfully")
        except Exception as e:
            logger.error(f"Error encoding docs: {e}")
            raise e
        return embeds

    def _initialize_faiss_index(self, dim: int):
        """
        Initialize the FAISS index with a given dimensionality for similarity search.

        Purpose:
        - This function initializes a FAISS index object for efficient similarity-based search.
        - It creates an index based on the inner product (IP) similarity metric, suitable for tasks like nearest neighbor search.
        - The function assigns the initialized index to `self.index`, which will later be used to store and search embeddings.

        Parameters:
        - dim (int): The dimensionality of the embeddings, i.e., the number of features in each vector.

        Outputs:
        - None. The function does not return any value but modifies the internal state by initializing the FAISS index.

        Notes:
        - The function also resets the previously created index if it exists. 
        - The `faiss.IndexFlatIP` index is a simple and efficient index that uses inner product similarity.
        - The index is created on the CPU and will be populated later with embeddings.

        """
        self.index = None
        cpu_index = faiss.IndexFlatIP(dim)
        self.index = cpu_index
        return

    def _move_index_to_gpu(self):
        """
        Move the FAISS index from the CPU to multiple GPUs for faster similarity search.

        Purpose:
        - This function transfers the FAISS index from the CPU to one or more GPUs to enable faster computation for similarity-based search.
        - It leverages FAISS's GPU capabilities to utilize the parallel processing power of multiple GPUs for indexing and querying.

        Steps:
        1. **Determine available GPUs:** The function first checks how many GPUs are available using `faiss.get_num_gpus()`.
        2. **Create GPU resources:** It then initializes GPU resources (`faiss.StandardGpuResources()`) for each available GPU.
        3. **Configure GPU options:** The `faiss.GpuMultipleClonerOptions()` is configured, with sharding enabled to distribute the index across GPUs, and `usePrecomputed` set to `False` to prevent using precomputed data.
        4. **Move the index to GPUs:** Using `faiss.index_cpu_to_gpu_multiple()`, the function moves the index from the CPU to the GPUs, using the resources and options defined earlier.

        Parameters:
        - None. The function operates on the internal state of the class, specifically transferring `self.index` from the CPU to the GPU(s).

        Outputs:
        - None. The function modifies the internal state by moving the FAISS index to the GPU, improving performance for subsequent similarity searches.

        Notes:
        - This function requires that the FAISS library is available and configured to support GPU usage.
        - It uses multiple GPUs if available, distributing the index across them for scalability and efficiency.
        - This is beneficial for large-scale similarity searches that would benefit from the parallel processing capabilities of GPUs.

        """
        logger.info("Moving index to GPU")
        ngpu = faiss.get_num_gpus()
        gpu_resources = []
        for i in range(ngpu):
            res = faiss.StandardGpuResources()
            gpu_resources.append(res)
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        self.index = faiss.index_cpu_to_gpu_multiple(vres, vdev, self.index, co)
        return

    def init_index_and_add(self, embeds):
        """
        Initialize and populate the FAISS index with embeddings for efficient similarity search.
        Our current implementation uses FIASS index for fast similarity/semantic search.
        Purpose:
        - This function is responsible for setting up the FAISS index, a data structure used for fast nearest-neighbor searches.
        - It initializes the index based on the dimensionality of the provided embeddings, adds the embeddings to the index,
        and moves the index to the GPU if required for better performance.
        
        Steps:
        1. **Initialize FAISS index:** The function first determines the dimensionality of the embeddings and initializes
        the FAISS index accordingly.
        2. **Add embeddings to index:** The embeddings are then added to the index for efficient similarity-based queries.
        3. **Move to GPU (if needed):** If GPU usage is enabled (`self.use_gpu`), the index is transferred to the GPU for faster processing.

        Parameters:
        - embeds (numpy.ndarray): A 2D array of embeddings, where each row is an embedding vector corresponding to a document.

        Outputs:
        - None. The function does not return any value, but it modifies the internal state by initializing and populating the index.

        Notes:
        - This function requires that the FAISS library is available and the `self.index` is a FAISS index object.
        - Moving the index to the GPU is optional and depends on the value of the `self.use_gpu` flag.

        """
        
        logger.info("Initialize the index...")
        dim = embeds.shape[1]
        self._initialize_faiss_index(dim)
        self.index.add(embeds)

        if self.use_gpu:
            self._move_index_to_gpu()
        return

    @classmethod
    def build_embeddings(cls, model, corpus_dataset, args):
        """
        This function is unused in current Implementation
        """
        retriever = cls(model, corpus_dataset, args)
        retriever.doc_embedding_inference()
        return retriever

    @classmethod
    def from_embeddings(cls, model, args):
        """
        This function is unused in current Implementation
        """
        retriever = cls(model, None, args)
        if args.process_index == 0:
            retriever.init_index_and_add()
        if args.world_size > 1:
            torch.distributed.barrier()
        return retriever

    def reset_index(self):
        """
        This function is unused in current Implementation
        """
        if self.index:
            self.index.reset()
        self.doc_lookup = []
        self.query_lookup = []
        return

    def search_single(self, query, topk: int = 10):
        """
        Perform a similarity search for a single query and retrieve the top-k most relevant results.

        Purpose:
        - This function takes a textual query, embeds it into a vector using the model, and then searches the FAISS index
        for the top-k most similar items based on the query's embedding.
        - It returns the corresponding results from the graph along with their indices and similarity scores.

        Parameters:
        - query (str): A textual query that needs to be searched against the indexed graph.
        - topk (int, optional): The number of top results to return. Default is 10.

        Outputs:
        - result (list): A list of graph entries (documents) that are most similar to the query based on the FAISS index.
        - original_indice (list): A list of indices corresponding to the most relevant results in the graph.
        - scores (list): A list of similarity scores for the top-k results.

        Notes:
        - The function first checks whether the index has been initialized. If not, it raises a `ValueError`.
        - The query is embedded using the `embed_query` function, and the similarity search is performed using the FAISS index.
        - The function returns the graph entries corresponding to the top-k results, along with their indices and similarity scores.
        """
        # logger.info("Searching")
        if self.index is None:
            raise ValueError("Index is not initialized")
        
        query_embed = self.embed_query(query)

        D, I = self.index.search(query_embed[None,:], topk)
        # original_indice = np.array(self.doc_lookup)[I].tolist()[0][0]
        # original_type = np.array(self.doc_type)[I].tolist()[0][0]

        original_indice = np.array(self.doc_lookup)[I].tolist()[0]
        original_type = np.array(self.doc_type)[I].tolist()[0]
        scores = D.tolist()[0]

        # return original_indice, self.graph[f'{original_type}_nodes'][original_indice]

        result = []
        for index in original_indice:
            # row_num = index // len(self.graph[0])
            # col_num = index % len(self.graph[0])
            result.append(self.graph[index])
        return result,original_indice,scores


    def search_with_dynamic_k_and_silhoutee_score(self, query, min_threshold_k=3, max_threshold_k=50):
        """
        Searches for the most relevant results in a FAISS index using dynamic clustering 
        (Silhouette Coefficient-based optimal k-means clustering) and returns a subset 
        of results based on the largest cluster.

        Parameters:
        ----------
        query : Any
            The input query to search for. It is processed to generate embeddings 
            compatible with the FAISS index.
        max_threshold_k : int, optional, default=50
            The maximum number of results to return after clustering and filtering.

        min_threshold_k : int, optional, default=3
        Returns:
        -------
        selected_result : list
            A list of results corresponding to the largest cluster, sorted in descending 
            order of similarity scores.
        selected_original_indice : list
            A list of indices in the original data that correspond to the selected results.
        selected_scores : list
            A list of similarity scores corresponding to the selected results.
        
        Function Workflow:
        ------------------
        1. Retrieve all potential matches (`result`, `original_indice`, `scores`) for the input `query` 
        by searching the entire FAISS index.
        2. Perform a deep copy of the `scores` for further processing.
        3. Use k-means clustering with a dynamic `k` to identify clusters in the data. 
        The optimal value of `k` is determined using the Silhouette Coefficient.
        - The Silhouette Coefficient evaluates how well-separated clusters are.
        - Start clustering with `init_k=2` (minimum clusters required for Silhouette analysis) 
            and go up to `max_k=topk`.
        4. Identify the largest cluster based on the centroid values.
        5. Filter the results, retaining only those belonging to the largest cluster.
        6. Sort the filtered results based on similarity scores in descending order.
        7. Select the top `max_threshold_k` results (or fewer if fewer elements are in the cluster).
        8. Return the selected results, original indices, and similarity scores.

        Notes:
        ------
        - The function assumes that `self.search_single` is implemented and returns:
            result: List of search results.
            original_indice: List of indices in the original data for the results.
            scores: List of similarity scores for the results.
        - The use of `np.argmax` ensures that the largest cluster is correctly identified 
        based on centroid magnitudes.
        """
         
        topk = self.index.ntotal
        result,original_indice,scores = self.search_single(query, topk)

        # Deep copy the scores list
        data = np.array(deepcopy(scores))
        data = data.reshape(-1, 1)
        
        silhouette_scores = []
        k_values = []
       
        # we have set init_k=2 because SILHOUETTE COEFFICIENT needs atleast two clusters to begin its processing.
        init_k = 2
        max_k = topk

        # Loop the range of k values to test
        for k in range(init_k, max_k):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, labels)
            silhouette_scores.append(silhouette_avg)
            k_values.append(k)
        
        # Find the optimal k value based on the maximum silhouette score
        optimal_k = k_values[np.argmax(silhouette_scores)]
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        labels = kmeans.fit_predict(data)

        # Get cluster labels and centroids
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        largest_centroid_index = np.argmax(centroids)
        selected_scores = []
        selected_original_indice = []
        selected_result = []

        for point,label in zip (data, labels):
            if (len(selected_scores)>=max_threshold_k):
                break
            if (label==largest_centroid_index):
                selected_scores.append(point)

        sorted_data = sorted(zip(scores, original_indice, result), key=lambda x: x[0], reverse=True)
        for sc, orig_ind, res in sorted_data:
            if len(selected_result) >= max_threshold_k and len(selected_original_indice)>=max_threshold_k:
                break
            if sc in selected_scores:
                selected_original_indice.append(orig_ind)
                selected_result.append(res)
        
        
        for sc, orig_ind, res in sorted_data:
            if (len(selected_result)> min_threshold_k or len(selected_original_indice) > min_threshold_k):
                break
            elif ((sc not in selected_scores) and 
                (orig_ind not in selected_original_indice) and 
                (res not in selected_result)
                ):
                selected_result.append(res)
                selected_original_indice.append(orig_ind)
                selected_scores.append(sc)


        return selected_result,selected_original_indice,selected_scores


def load_dense_retriever(args):
    node_retriever=Retriever(args)

    return node_retriever


