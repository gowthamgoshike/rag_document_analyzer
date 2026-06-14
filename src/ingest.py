import os
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

def extract_text_from_pdf(pdf_path):
    """Extracts all raw text from a given PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Document not found at: {pdf_path}")
    
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"
    return raw_text

def chunk_text(text, chunk_size=600, chunk_overlap=100):
    """Splits text into overlapping chunks based on character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def load_into_vector_db(chunks, db_path="./chroma_db"):
    """Embeds and saves text chunks into a local serverless ChromaDB."""
    # Initialize the client pointing to a local directory
    client = chromadb.PersistentClient(path=db_path)
    
    # Use a lightweight, local embedding function running via PyTorch
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create or fetch the collection
    collection = client.get_or_create_collection(
        name="document_analyzer", 
        embedding_function=embedding_func
    )
    
    # Prepare data for batch insertion
    ids = [f"id_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "uploaded_document"} for _ in chunks]
    
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Successfully ingested and indexed {len(chunks)} text chunks.")

if __name__ == "__main__":
    # Place a sample PDF inside src/data/sample.pdf to run this directly
    sample_pdf = "src/data/sample.pdf"
    
    if os.path.exists(sample_pdf):
        print("Extracting document text...")
        raw_document_text = extract_text_from_pdf(sample_pdf)
        
        print("Creating text chunks...")
        document_chunks = chunk_text(raw_document_text)
        
        print("Initializing database and generating embeddings...")
        load_into_vector_db(document_chunks)
    else:
        print(f"Please place a PDF at '{sample_pdf}' to test the ingestion script.")