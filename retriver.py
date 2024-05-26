from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from qdrant_client import QdrantClient

embeddings = SentenceTransformerEmbeddings(model_name = "NeuML/pubmedbert-base-embeddings")
# embeddings = OllamaEmbeddings(model = "llama3")

url = "http://localhost:6333/dashboard"

client = QdrantClient(
    url = url,
    prefer_grpc = False
)

print(client)

db = Qdrant(client=client,embeddings=embeddings,collection_name='vector_database')


print(db)
print("#############################")

query = "What are the common sided effect of systemic therapeutic agents?"

docs = db.similarity_search_with_score(query=query,k=2)

for i in docs:
    doc,score = i
    print({"score":score,"Content":doc.page_content,"metadata":doc.metadata})