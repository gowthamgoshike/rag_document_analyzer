# Local Decoupled RAG Document Analyzer

A production-grade, privacy-first Retrieval-Augmented Generation (RAG) pipeline built entirely on native macOS architecture without external package managers. This system features a decoupled microservice layout, separating the backend ingestion and LLM orchestration layer from the frontend user interface. It runs 100% locally to ensure total data compliance and zero API costs.

## 🏗️ System Architecture

The application is engineered as a distributed system to ensure scalability and clear separation of concerns:

1. **Ingestion Pipeline:** Extracts unstructured text from multi-format documents (`.pdf`, `.docx`), chunks the text with contextual overlap, and maps them into a semantic vector space.
2. **Vector Database:** Utilizes a persistent, serverless instance of ChromaDB to index dense vector embeddings generated locally via PyTorch. Telemetry is explicitly disabled for privacy and performance.
3. **Backend REST API (FastAPI):** Exposes endpoints for processing user prompts, orchestrates vector similarity queries, constructs context-grounded system prompts, and interfaces with local LLMs.
4. **Frontend UI Engine (Streamlit):** A native Python conversational interface that manages stateful chat history and communicates asynchronously with the API gateway.

---

## 🛠️ Tech Stack & Tooling

* **Language:** Python 3.11+ (Native macOS Environment)
* **API Framework:** FastAPI / Uvicorn
* **UI Framework:** Streamlit
* **Vector Database:** ChromaDB (Persistent Serverless Client)
* **Embedding Model:** HuggingFace `all-MiniLM-L6-v2` via Sentence-Transformers
* **Local Inference Engine:** Ollama (Llama 3 Model)
* **Document Parsers:** PyPDF & Python-Docx

---

## 📁 Repository Structure

```text
rag_document_analyzer/
├── src/
│   ├── data/
│   │   └── sample.pdf           # Ingested mock source documents
│   ├── ingest.py                # Document extraction, chunking, and DB loading
│   ├── query.py                 # Pure CLI retrieval & generation module
│   ├── main.py                  # FastAPI high-performance application server
│   └── app.py                   # Streamlit interactive user interface
├── .gitignore                   # Strict path exclusions (DB & Venv blocks)
└── requirements.txt             # Deterministic dependency manifest
```

---

## 🚀 Step-by-Step Installation & Setup

This repository is optimized for macOS setups using built-in system tools directly.

### 1. Environment Initialization
Clone this repository to your local directory and initialize an isolated virtual environment:
```bash
git clone [https://github.com/YOUR_USERNAME/rag-document-analyzer.git](https://github.com/YOUR_USERNAME/rag-document-analyzer.git)
cd rag_document_analyzer
python3 -m venv ragvenv
source ragvenv/bin/activate
```

### 2. Install Deterministic Dependencies
Install the required system packages and pinned model constraints inside your active environment:
```bash
pip install -r requirements.txt
```

### 3. Setup Local Inference (Ollama)
1. Download the native desktop application directly from [ollama.com/download](https://ollama.com/download).
2. Install the app to your macOS `Applications` directory and launch it.
3. Pull the Llama 3 foundation weights by running the following command in your terminal:
   ```bash
   ollama pull llama3
   ```

---

## 🏃 Execution Instructions

### Phase 1: Document Ingestion & Indexing
Place your sample target files into the `src/data/` directory. Run the ingestion module to extract, chunk, and embed the raw text into your local vector database:
```bash
python src/ingest.py
```
*Note: This creates a persistent binary folder named `chroma_db/` in your workspace directory, which is excluded from remote version control.*

### Phase 2: Launch the Microservice Infrastructure
Because this is a decoupled application, you must run the backend engine and frontend client inside two separate terminal sessions with your virtual environment active.

**Terminal 1: Start the FastAPI Engine**
Force the interpreter to execute Uvicorn directly from your local environment binary path to avoid path bleeding:
```bash
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2: Start the Streamlit Interface Client**
Open a new terminal window, reactivate the project environment, and trigger the web renderer:
```bash
source ragvenv/bin/activate
streamlit run src/app.py
```

The interface will initialize and automatically open in a new tab within your default web browser at `http://localhost:8501`.

---

## 🔒 Production Defensiveness & Edge Case Handling

* **Dependency Conflict Prevention:** Expressly binds `numpy<2.0.0` to eliminate fatal API errors with legacy ChromaDB tensor translations.
* **Telemetry Silencing:** Explicitly configures `Settings(anonymized_telemetry=False)` on database drivers to halt background tracking overhead and prevent subprocess crashes.
* **Port Collision Avoidance:** Explicitly targets port `8000` for the REST API interface to bypass standard macOS process locks.
