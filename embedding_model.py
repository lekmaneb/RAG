import warnings
warnings.filterwarnings('ignore')

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document




model_encode_name = "sentence-transformers/all-MiniLM-L6-v2"
# model_encode_name = "mixedbread-ai/mxbai-embed-large-v1"
embedding_model = HuggingFaceEmbeddings(model_name=model_encode_name)