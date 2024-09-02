import copy
from typing import Optional, List

import numpy as np

from rag.llm_client import MistralLLMClient
from faiss import IndexFlatL2


class MistralRAG:
    def __init__(self):
        self.vector_store: Optional[IndexFlatL2] = None
        self.data_store: Optional[List[str]] = None
        self.llm_client: MistralLLMClient = MistralLLMClient()

    def load_index(self, index_path: str) -> None:
        pass

    def add_csv(self, doc_path: str) -> None:
        pass

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

    def query(self, query: str) -> str:
        pass

