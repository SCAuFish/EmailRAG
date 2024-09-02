# Mistral config
from typing import List

MISTRAL_API_KEY = "Q7p35bsjJqlWFy6SMCOgJHAkp8J2hH8q"
MISTRAL_MODEL = "mistral-large-latest"
MISTRAL_EMBED_MODEL = "mistral-embed"

# Asset config
INDEX_SAVE_PATH = "./assets/index.pkl"

# App-related config
QUERY_AUGMENTATION_SYSTEM_PROMPT = \
    ("Please draft an business email that might address the input topic or question. "
     "Please keep things short, concise yet professional. Skip subject, opening and ending courtesies.")
RELEVANT_DOC_COUNT = 5
QA_SYSTEM_PROMPT = ("You are a helpful assistant who could help find the right answer to the question with "
                    "given information. You only need to provide the answer with very concise reference to the "
                    "information that helps generate the answer.")


def document_augmented_query(question: str, retrieved_docs: List[str]):
    return (f"Question: {question}\n"
            f"Here are some useful reference documents:\n"
            "\n".join(retrieved_docs))
