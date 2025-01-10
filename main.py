from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

model_llm_name = "microsoft/Phi-3.5-mini-instruct"
model_llm = AutoModelForCausalLM.from_pretrained(
    model_llm_name,
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer_llm = AutoTokenizer.from_pretrained(model_llm_name)

model_encoder_name = "dunzhang/stella_en_400M_v5"
model = SentenceTransformer(model_encoder_name, trust_remote_code=True).cuda()

query_prompt_name = "s2p_query"
queries = [
    "What does Malek like?",
]

# import requests
# from bs4 import BeautifulSoup

# def wiki_text_extractor(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     wiki_text = ''
#     for paragraph in soup.find_all('p'):
#         wiki_text += paragraph.get_text()
#     return wiki_text

# text = wiki_text_extractor("https://en.wikipedia.org/wiki/World_War_II")

# from nltk.tokenize import sent_tokenize

# # Diviser le texte en phrases
# sentences = sent_tokenize(text)
# print('sentence tokenized done')
# # Liste des phrases (output utilisable dans un système RAG)
# # splitted_sentences = list(sentences)
# splitted_sentences = []
# for sentence in sentences:
#     splitted_sentences.extend([sentence[i:i + 32] for i in range(0, len(sentence), 32)])
import PyPDF2
import sys

def pdf_text_extractor(pdf_file):
    pdf_file = open(pdf_file, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pdf_text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()
    pdf_file.close()
    return pdf_text

text = pdf_text_extractor("pdf.pdf")

from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(text)

# Liste des phrases (output utilisable dans un système RAG)
# split with max length of 128 tokens

splitted_sentences = []
for sentence in sentences:
    splitted_sentences.extend([sentence[i:i + 32] for i in range(0, len(sentence), 32)])


splitted_sentences.append('Malek loves cats.')

docs = splitted_sentences[-50:]
print('encoding...')
query_embeddings = model.encode(queries, prompt_name=query_prompt_name)
doc_embeddings = model.encode(docs)
print('encoded')

print('similarity...')
similarities = model.similarity(query_embeddings, doc_embeddings)
print('similarity done')

# get the index of the 15 most similar sentences

top_k = 5
top_k_idx = (-similarities.cpu().numpy()).argsort()[:top_k]

tok_k_idx_bis = top_k_idx[0]

# store the 15 most similar sentences
top_k_sentences = [docs[i] for i in tok_k_idx_bis]

# del encoder model
del model
del doc_embeddings
del query_embeddings
del similarities

messages = [
    {"role": "system", "content": f"You have this context : {top_k_sentences}; answer the following question but you can only use the context i gave you before, not your informations."},
    {"role": "user", "content": "What does Malek like?"},
]

pipe = pipeline(
    "text-generation",
    model=model_llm,
    tokenizer=tokenizer_llm,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}
print('generation...')
output = pipe(messages, **generation_args)
print(output[0]['generated_text'])