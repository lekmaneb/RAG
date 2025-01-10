import phi
import stella

query_prompt_name = "s2p_query"
queries = [
    "What does Malek like?",
]

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

docs = splitted_sentences
print('encoding...')
query_embeddings = stella.model_encode.encode(queries, prompt_name=query_prompt_name)
doc_embeddings = stella.model_encode.encode(docs)
print('encoded')

print('similarity...')
similarities = stella.model_encode.similarity(query_embeddings, doc_embeddings)
print('similarity done')

# get the index of the 15 most similar sentences

top_k = 15
top_k_idx = (-similarities.cpu().numpy()).argsort()[:top_k]

tok_k_idx_bis = top_k_idx[0]

# store the 15 most similar sentences
top_k_sentences = [docs[i] for i in tok_k_idx_bis]

# # del encoder model
# del model
# del doc_embeddings
# del query_embeddings
# del similarities

messages = [
    {"role": "system", "content": f"You have this context : {top_k_sentences}; answer the following question but you can only use the context i gave you before, not your informations."},
    {"role": "user", "content": "What does Malek like?"},
]

from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=phi.model_llm,
    tokenizer=phi.tokenizer_llm,
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