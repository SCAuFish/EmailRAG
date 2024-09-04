# Introduction
An AI assistant that helps navigate information in the ocean of emails leveraging Mistral AI API, 
and FAISS vector storage.
Some sample data are included for testing purpose. The data is a subset of 
[Enron email public dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset).

Overall design looks like below
![overview.jpeg](overview.jpeg)
# Installation and Quick Start
```bash
# Clone current repo
git clone https://github.com/SCAuFish/EmailRAG.git
cd EmailRAG
# Install required packages
poetry install
```
There are four operations supported in command line interface: bulk-add new emails in a jsonl file, add single email to 
the database, quit, or simply ask questions. Examples can be found below
```text
% myrag
2024-09-02 14:20:15,991 [rag.load_index:44] WARNING: Did not find index at path ./assets/index.pkl. Skipped loading
Welcome to Your Email Assistant! What would you like to do? Supported actions below
inc [filename]: Configure a jsonl file with your emails you'd like me to help with. Each line should contain 'from', 'to', and 'content' fields.
add [from] [to] [email content]: Share a new email that you'd like me to help with.
quit: Quit the program.
[Ask any question]: ask any question that are related to the emails.

(0) > inc assets/mini_sample_emails.jsonl

(30) > phone number of Kim?
Kim's phone number is 503-805-2117. (Reference: Email from Kim Ward to Phillip K Allen)
Most relevant documents: [3, 12, 21, 6, 10]

(30) > add Cheng Shen->Joe Biden: Looking forward to meeting you this Friday at 10AM!        

(31) > when's the interview with Biden?
The interview with Biden is on Friday at 10 AM. (Reference: Email from Cheng Shen to Joe Biden)
Most relevant documents: [30, 4, 17, 25, 3]

(31) > quit
```

# Future Developments
## Interface and Features
1. **GUI** Current command line interface could be further optimized with graphical interface.
2. **Auto-refresh** For now, we rely on the user to add new jsonl files containing emails to start indexing. Ideally the system should be
able to automatically pull emails from email server
3. **User Feedback** Add a feature to collect users satisfaction with an answer and their actions afterwards to help 
improve the model.

## Pre-retrieval
1. **More Powerful Indexing** Current implementation is using plain Mistral-embed model on the whole email content, 
regardless of email length, email format. Sender/recipient information is also embedded into the string as a prefix, rather
than leveraging them more systematically. It might lead to some information loss.
2. **Customized Embedding Model** Based on the use case, a customized embedding model could more accurately identify 
paraphrased sentences and terms. For example, a company might use multiple codenames for one project at different stage,
where all the codenames should be treated as equivalent.

## Retrieval
1. **Retrieval Accuracy** Current implementation leverages the HyDE trick (hypothetical document embedding), by getting a tentative answer from LLM, 
and using the answer as search query to find matching keys. There are other ways to improve retrieval accuracy, such as
retrieved result reranking, more fine-grained chunking strategies, or other query transformations.
2. **Retrieval Efficiency** 
   1. Some tricks above would take longer to execute. HyDE, for example, involves one more trigger
   of LLM. To speed things up, different tricks could run in parallel. And we compare the final results.
   2. Leverage hierarchical retrieval to first identify relevant emails of specific senders/recipients/topics, and do more
   fine-grained search.

## Augmentation
1. **Augmentation Rewriting** Currently the whole retrieved emails are included in the prompt as relevant information. 
It could include many irrelevant distractions.

## Generation
1. **SFT and RLHF** Leverage annotated high-quality data, or user feed back to find the output that meet users' needs the
most.

## Scalability
1. **Semantic Caching** Use semantic caching so that for questions of same meanings, we can directly return the results.
2. **Service Deployment of Each Module** Each of the module could be deployed as their own service, so that it's easier to 
maintain and upgrade, and is able to handle more concurrent requests.