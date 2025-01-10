from sentence_transformers import SentenceTransformer

model_encode_name = "dunzhang/stella_en_400M_v5"
model_encode = SentenceTransformer(model_encode_name, trust_remote_code=True).cuda()
query_prompt_name = "s2p_query"

queries = [
    "What does Malek like?",
]

import requests
from bs4 import BeautifulSoup

def wiki_text_extractor(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    wiki_text = ''
    for paragraph in soup.find_all('p'):
        wiki_text += paragraph.get_text()
    return wiki_text

text = wiki_text_extractor("https://www.apple.com/ios/ios-18/")

from nltk.tokenize import sent_tokenize

sentences = sent_tokenize(text)