from typing import List, Optional

from mistralai import ChatCompletionResponse, Mistral

from config import MISTRAL_API_KEY, MISTRAL_EMBED_MODEL, MISTRAL_MODEL


class MistralLLMClient:
    def __init__(self):
        super().__init__()
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.model = MISTRAL_MODEL

    def query(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = [
            {
                "role": "user", "content": user_prompt,
            }
        ]
        if system_prompt:
            messages.append({
                "role": "system", "content": system_prompt,
            })

        response: Optional[ChatCompletionResponse] = self.client.chat.complete(
            model=self.model,
            messages=messages,
        )

        return response.choices[0].message.content

    def embeddings(self, inputs: List[str]) -> List[List[float]]:
        return [
            d.embedding for d in self.client.embeddings.create(
                inputs=inputs,
                model=MISTRAL_EMBED_MODEL
            ).data
        ]

