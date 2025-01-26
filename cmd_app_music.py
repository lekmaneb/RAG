import warnings
warnings.filterwarnings('ignore')

import llm_model
import embedding_model
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import sys
import time
import pandas as pd

#open music_data.csv
music_df = pd.read_csv('music_data/music_data.csv')

splits = []
for idx, row in music_df.iterrows():
    splits.append(f"Name of the artist: {row['Artist']}; Name of the song: {row['Title']}; Lyrics of the song: {row['Lyric']}")

documents = [Document(page_content=split) for split in splits]

# remove all punctuation from the link to use it as a directory name
if embedding_model.model_encode_name == "sentence-transformers/all-MiniLM-L6-v2":
    persist_directory = "docs/chroma_small/music_data"
elif embedding_model.model_encode_name == "mixedbread-ai/mxbai-embed-large-v1":
    persist_directory = "docs/chroma/music_data"
elif embedding_model.model_encode_name == "ibm-granite/granite-embedding-125m-english":
    persist_directory = "docs/chroma_medium/music_data"
else:
    print("Please use a valid model name")
    sys.exit()

# check if the vector database already exists
if os.path.exists(persist_directory):
    # If the folder exists, we attempt to load the existing DB
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model.embedding_model
    )
    print(f"Loaded existing Chroma DB from {persist_directory}")
else:
    # Otherwise, we create a new DB from the documents
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model.embedding_model,
        persist_directory=persist_directory
    )
    vectordb.persist()  # Persist to disk
    print(f"Created and saved new Chroma DB at {persist_directory}")

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Answer only by giving the name of the song and the artist of the song you choosed. 
{context}
Question: {question}
Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain

qa_chain = RetrievalQA.from_chain_type(
    llm_model.llm,
    return_source_documents=True,
    retriever=vectordb.as_retriever(search_kwargs={"k": 10}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

while True:

    print("\n")
    print("Enter your question: ")
    question = input()
    print("\n")

    while question == "":
        print("Please enter a valid question")
        question = input()
        print("\n")

    result = qa_chain({"query": question})

    #show only the answer, not context before it

    answer = result["result"]
    answer = answer.split("Answer: ")[1]

    # when you print answer, animate the text to show that the answer is being printed
    def animate_text(text):
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.01)  # Adjust the delay as needed
        print()

    print('Answer:')
    animate_text(answer)