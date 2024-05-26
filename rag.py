from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import os
import json
from huggingface_hub import login
import logging

# from langchain_community.embeddings import OllamaEmbeddings


# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Hugging Face login
try:
    login(token="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")
    logging.debug("Hugging Face login successful.")
except Exception as e:
    logging.error(f"Error during Hugging Face login: {e}")

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub.file_download')

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

local_llm = "meditron-7b-chat.Q4_K_M.gguf"

config = {
    'max_new_tokens': 1024,
    'context_length': 2048,
    'repetition_penalty': 1.1,
    'temperature': 0.1,
    'top_k': 50,
    'top_p': 0.9,
    'stream': True,
    'threads': int(os.cpu_count() / 2)
}

try:
    llm = CTransformers(
        model=local_llm,
        model_type="llama",
        lib="avx2",
        **config
    )
    logging.debug("LLM Initialized.")
except Exception as e:
    logging.error(f"Error initializing LLM: {e}")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

try:
    embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    # embeddings = OllamaEmbeddings(model = "llama3")
    logging.debug("Embeddings model initialized.")
except Exception as e:
    logging.error(f"Error initializing embeddings model: {e}")

url = "http://localhost:6333"

try:
    client = QdrantClient(
        url=url, prefer_grpc=False
    )
    logging.debug("Qdrant client initialized.")
except Exception as e:
    logging.error(f"Error initializing Qdrant client: {e}")

try:
    db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_database")
    logging.debug("Qdrant database initialized.")
except Exception as e:
    logging.error(f"Error initializing Qdrant database: {e}")

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

try:
    retriever = db.as_retriever(search_kwargs={"k": 1})
    logging.debug("Retriever initialized.")
except Exception as e:
    logging.error(f"Error initializing retriever: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs,
            verbose=True
        )
        logging.debug("RetrievalQA initialized.")
        
        response = qa(query)
        logging.debug(f"Response generated: {response}")
        
        answer = response['result']
        source_document = response['source_documents'][0].page_content
        doc = response['source_documents'][0].metadata['source']
        
        response_data = jsonable_encoder(json.dumps({"answer": answer, "source_document": source_document, "doc": doc}))
        res =  Response(content=response_data)
        return res
    except Exception as e:
        logging.error(f"Error during query processing: {e}")
        return Response(content="Internal Server Error", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    #http://localhost:6333/dashboard
