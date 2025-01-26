from flask import Flask, render_template, request, jsonify

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

music_df = pd.read_csv('music_data.csv')

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

app = Flask(__name__)

# Ici on conserve l'historique des messages dans une liste Python.
# Dans un vrai projet de conversation, vous pourriez utiliser une base de données
# ou stocker l'historique dans la session de l'utilisateur.
chat_history = []

@app.route("/")
def index():
    """
    Affiche la page HTML du chat.
    """
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    """
    Reçoit la question en JSON, exécute le modèle LLM/RAG et renvoie la réponse au format JSON.
    """
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question vide"}), 400

    # -------------------------------------------------------------------
    # 1) Placez ici l'appel à votre pipeline RAG / LLM, ex:
    result = qa_chain({"query": question})
    reponse_llm = result["result"]
    reponse_llm = reponse_llm.split("Answer: ")[1]
    #
    # Pour la démo, on va simplement renvoyer un texte statique.
    # -------------------------------------------------------------------
    # reponse_llm = f"Ceci est une réponse simulée pour la question: {question}"

    # On empile la question et la réponse dans l'historique local
    chat_history.append(("user", question))
    chat_history.append(("bot", reponse_llm))

    return jsonify({"answer": reponse_llm})

if __name__ == "__main__":
    app.run(debug=True)
