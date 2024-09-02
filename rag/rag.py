import copy
import json
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
from faiss import IndexFlatL2

from config import QUERY_AUGMENTATION_SYSTEM_PROMPT, RELEVANT_DOC_COUNT, document_augmented_query, QA_SYSTEM_PROMPT, \
    INDEX_SAVE_PATH
from rag.llm_client import MistralLLMClient
from rag.utils import activate_logger

logger = activate_logger("rag")


class MistralRAG:
    def __init__(self):
        self.vector_store: Optional[IndexFlatL2] = None
        self.data_store: Optional[List[str]] = None
        self.llm_client: MistralLLMClient = MistralLLMClient()

        self.load_index()

    def load_index(self, index_path: str = INDEX_SAVE_PATH) -> None:
        if os.path.exists(index_path):
            loaded = pickle.load(open(index_path, "rb"))

            self.vector_store = loaded["vector_store"]
            self.data_store = loaded["data_store"]

            logger.info(f"Loaded index from {index_path}")

        logger.warning(f"Did not find index at path {index_path}. Skipped loading")

    def add_jsonl(self, doc_path: str) -> None:
        all_data = []
        with open(doc_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                all_data.append(f"Email from {data['from']} to {data['to']} below\n{data['content']}")

        self.add_contents(all_data)

    def save_index(self, save_path: str = INDEX_SAVE_PATH) -> None:
        to_save = {
            'vector_store': self.vector_store,
            'data_store': self.data_store,
        }
        with open(save_path, 'wb') as writer:
            pickle.dump(to_save, writer)

    def add_contents(self, contents: List[str]) -> None:
        embeddings: np.array = np.array(self.llm_client.embeddings(contents))

        # If no vector store loaded or created
        if self.vector_store is None:
            expected_dim = len(embeddings[0])

            self.vector_store = IndexFlatL2(expected_dim)
            self.vector_store.add(np.array(embeddings))

            self.data_store = copy.deepcopy(contents)

        else:
            self.vector_store.add(embeddings)
            self.data_store.extend(contents)

        self.save_index()

    def query(self, query: str) -> Tuple[str, List[int]]:
        # 1. Get query embedding. Augment query by fabricating a possible answer first.
        augmented_query = self.llm_client.query(user_prompt=query, system_prompt=QUERY_AUGMENTATION_SYSTEM_PROMPT)
        embedding = np.array(self.llm_client.embeddings([augmented_query]))

        # 2. Retrieve relevant document
        distances, indices = self.vector_store.search(embedding, k=RELEVANT_DOC_COUNT)

        # 3. Send query and get answer
        retrieved_indices = indices.tolist()[0]
        retrieved_docs = [self.data_store[index] for index in retrieved_indices]
        augmented_query = document_augmented_query(query, retrieved_docs)
        result = self.llm_client.query(user_prompt=augmented_query, system_prompt=QA_SYSTEM_PROMPT)

        return result, retrieved_indices
