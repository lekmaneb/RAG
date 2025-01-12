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

def text_extractor(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ''
    for paragraph in soup.find_all('p'):
        text += paragraph.get_text()
    return text

link = input("Enter the link of something: ")

while not link.startswith("http"):
    print("Please enter a valid link")
    link = input("Enter the link of something: ")

try:
    text = text_extractor(link)
except:
    print("Error while trying to extract text from the link, restart the program and try again")
    sys.exit()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)

splits = text_splitter.split_text(text)

documents = [Document(page_content=split) for split in splits]

# remove all punctuation from the link to use it as a directory name
persist_directory = f"docs/chroma/{link.replace('https://', '').replace('http://', '').replace('/', '')}"

# for small embedding model :
# persist_directory = f"docs/chroma_small/{link.replace('https://', '').replace('http://', '').replace('/', '')}"

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
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Don't give in any case an explanation of your answer. Don't give any additional information. Just answer the question. 
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