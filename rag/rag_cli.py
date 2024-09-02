# Main function providing CLI to user
import re

from rag.rag import MistralRAG
from rag.utils import construct_content

INCLUDE = 'inc'
ADD = 'add'
QUIT = 'quit'


def main():
    rag = MistralRAG()

    print("Welcome to Your Email Assistant! What would you like to do? Supported actions below")
    print(f"{INCLUDE} [filename]: Configure a jsonl file with your emails you'd like me to help with. "
          "Each line should contain 'from', 'to', and 'content' fields.")
    print(f"{ADD} [from] [to] [email content]: Share a new email that you'd like me to help with.")
    print(f"{QUIT}: Quit the program.")
    print("[Ask any question]: ask any question that are related to the emails.")
    while True:
        action = input(f"({rag.size()}) > ")
        if action.startswith(INCLUDE):
            filename = re.search(rf"{INCLUDE} (.*)", action).group(1)
            rag.add_jsonl(filename)
        elif action.startswith(ADD):
            from_name, to_name, email_content = re.search(rf"{ADD} (.*)->(.*): (.*)", action).groups()
            rag.add_contents([construct_content(from_name, to_name, email_content)])
        elif action.startswith(QUIT):
            break
        else:
            response, relevant_indices = rag.query(action)
            print(response)
            print(f"Most relevant documents: {relevant_indices}")


if __name__ == '__main__':
    main()