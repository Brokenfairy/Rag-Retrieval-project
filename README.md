# LangChain RAG Starter

Minimal retrieval-augmented generation (RAG) app using:
- LangChain
- Ollama (chat + embeddings)
- Milvus / Zilliz Cloud
- Streamlit UI

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Configure environment

Edit `.env`:

```env
# Zilliz Cloud Configuration (leave empty to use local Milvus Lite file)
ZILLIZ_CLOUD_URI=""
ZILLIZ_CLOUD_TOKEN=""

# LlamaParse Configuration (optional)
LLAMA_CLOUD_API_KEY=""

# Ollama Configuration
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="llama3"
OLLAMA_EMBED_MODEL="llama3"
MILVUS_COLLECTION="langchain_docs"
```

Notes:
- If `ZILLIZ_CLOUD_URI` is empty, the app defaults to local `data/milvus_lite.db`.
- For local use, make sure Ollama is running and the selected model exists.

## 3) Run the app

```bash
streamlit run app.py
```

## 4) Use the app

1. Upload files (`.txt`, `.md`, `.pdf`, `.csv`, `.msg`).
2. Click **Index uploaded files**.
3. Ask questions in chat.

## Project Structure

```
app.py
src/
  config.py
  ingestion.py
  models.py
  rag.py
  vectorstore.py
```

