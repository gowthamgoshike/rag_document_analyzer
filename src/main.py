from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import ollama

app = FastAPI(title="RAG Analyzer Engine")

# Define the expected JSON payload schema
class QueryRequest(BaseModel):
    prompt: str

# Re-use the identical local vector database setup
DB_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

@app.post("/query")
async def handle_query(request: QueryRequest):
    """Processes a user prompt, retrieves context from ChromaDB, and calls Ollama."""
    try:
        # 1. Fetch the collection
        collection = client.get_collection(name="document_analyzer", embedding_function=embedding_func)
        
        # 2. Retrieve relevant context chunks
        results = collection.query(
            query_texts=[request.prompt],
            n_results=3
        )
        retrieved_chunks = results['documents'][0]
        context = "\n---\n".join(retrieved_chunks)
        
        # 3. Build the grounded prompt
        system_prompt = (
            "You are an expert document analyzer. Answer the user's question accurately using ONLY "
            "the provided context below. If the context does not contain the answer, explicitly state "
            "that you do not have enough information to answer.\n\n"
            f"Context:\n{context}"
        )
        
        # 4. Generate the response via local Ollama (non-streaming for API simplicity)
        response = ollama.chat(
            model='llama3',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': request.prompt}
            ]
        )
        
        return {
            "answer": response['message']['content'],
            "context_chunks_retrieved": len(retrieved_chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))