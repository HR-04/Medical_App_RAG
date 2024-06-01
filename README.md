# Meditron Chatbot [Medical RAG App]

## Description

Meditron Chatbot is a Medical Retrieval-Augmented Generation (RAG) application designed to provide relevant answers along with page numbers and context. The chatbot is built to avoid hallucinations, ensuring accurate and reliable responses. 
It leverages an open-source instruct Large Language Model (LLM), embeddings, and a vector store that operates locally.

## Tech Stack

### Front End
- **HTML**
- **Bootstrap**

### Back End
- **Flask**
- **FastAPI**

### Model
- **Meditron**

### Embeddings
- **Hugging Face Instruct Embeddings**

### Vector Store
- **Qdrant** (running using Docker)

### Framework
- **Langchain**
- **CTransformers** (as orchestration frameworks)

## Installation

### Prerequisites
- Docker
- Python 3.8+
- Flask
- FastAPI

### Steps
1. **Clone the Repository**
    ```sh
    git clone https://github.com/yourusername/meditron-chatbot.git
    cd meditron-chatbot
    ```

2. **Set Up the Environment**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4. **Run Qdrant using Docker**
    ```sh
    docker run -p 6333:6333 -d qdrant/qdrant
    ```

5. **Start the Back End**
    ```sh
    # In one terminal
    flask run

    # In another terminal
    uvicorn app:rag.app --reload or python rag.py
    ```

6. **Open the Front End**
    - Open `index.html` in your browser.

## Usage
1. Open the web application in your browser.
2. Enter your medical query into the chat interface.
3. Receive responses with relevant information, including page numbers and context.

## Contributing
We welcome contributions! Please fork the repository and submit a pull request.


