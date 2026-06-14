import chromadb
from chromadb.utils import embedding_functions
import ollama

def query_rag_system(user_query, db_path="./chroma_db", num_results=3):
    """Retrieves relevant text chunks and answers the user query using Llama 3."""
    client = chromadb.PersistentClient(path=db_path)
    
    # Must use the identical embedding model used during ingestion
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    try:
        collection = client.get_collection(name="document_analyzer", embedding_function=embedding_func)
    except Exception:
        print("Database collection not found. Please run ingest.py first.")
        return

    # Retrieve matching items
    results = collection.query(
        query_texts=[user_query],
        n_results=num_results
    )
    
    retrieved_chunks = results['documents'][0]
    
    # Combine chunks to form the context block
    context = "\n---\n".join(retrieved_chunks)
    
    # Construct a strict prompt to ground the LLM
    system_prompt = (
        "You are an expert document analyzer. Answer the user's question accurately using ONLY "
        "the provided context below. If the context does not contain the answer, explicitly state "
        "that you do not have enough information to answer.\n\n"
        f"Context:\n{context}"
    )
    
    print("\n--- Sending Augmented Prompt to Local Llama 3 ---")
    
    # Stream the response directly to the terminal
    response_stream = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query}
        ],
        stream=True
    )
    
    for chunk in response_stream:
        print(chunk['message']['content'], end='', flush=True)
    print("\n")

if __name__ == "__main__":
    # Ensure your Ollama application is running in the background before execution
    query = input("Ask a question about your ingested document: ")
    if query.strip():
        query_rag_system(query)