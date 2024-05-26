from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub.file_download')

# embeddings = OllamaEmbeddings(model = "llama3")
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

print(embeddings)

loader=DirectoryLoader('Data/',glob='**/*.pdf',show_progress =True,loader_cls=PyPDFLoader)

document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)

texts = text_splitter.split_documents(document)

url = "http://localhost:6333/dashboard"

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url = url,
    prefer_grpc = False,
    collection_name = 'vector_database'
)

print("vector Database is Created")

# run docker run -p 6333:6333 qdrant/qdrant


