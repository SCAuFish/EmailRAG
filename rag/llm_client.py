from abc import ABC, abstractmethod
from typing import Optional, List

from config import MISTRAL_API_KEY, MISTRAL_MODEL, MISTRAL_EMBED_MODEL
from mistralai import Mistral, ChatCompletionResponse


class MistralLLMClient:
    def __init__(self):
        super().__init__()
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.model = MISTRAL_MODEL

    def query(self, query: str) -> str:
        response: Optional[ChatCompletionResponse] = self.client.chat.complete(
            model=self.model,
            messages="",
        )

        return ""

    def embeddings(self, inputs: List[str]) -> List[List[float]]:
        return [
            d.embedding for d in self.client.embeddings.create(
                inputs=inputs,
                model=MISTRAL_EMBED_MODEL
            ).data
        ]

