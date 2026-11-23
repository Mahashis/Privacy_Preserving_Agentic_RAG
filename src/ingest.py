"""
PDF ingestion pipeline for Privacy-Preserving Agentic RAG.

Loads PDFs, splits them into overlapping chunks, embeds them locally via Ollama
(nomic-embed-text), and writes them to ChromaDB.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from config_loader import config


def load_pdfs(data_dir: Path) -> List:
    """Load all PDF files from the data directory."""
    documents = []
    for pdf_path in data_dir.glob("*.pdf"):
        print(f"Loading {pdf_path.name}...")
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())
    return documents


def split_documents(documents: List) -> List:
    """Split documents into overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        separators=config.text_splitter_separators,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vectorstore(chunks: List, persist_dir: Path) -> Chroma:
    """Create and persist ChromaDB vector store with local embeddings."""
    embeddings = OllamaEmbeddings(model=config.embedding_model)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    print(f"Vector store created at {persist_dir}")
    return vectorstore


def main():
    """Main ingestion pipeline."""
    print("Starting PDF ingestion pipeline...")
    
    # Ensure directories exist
    config.data_dir.mkdir(exist_ok=True)
    config.chroma_dir.mkdir(exist_ok=True)
    
    # Load PDFs
    documents = load_pdfs(config.data_dir)
    if not documents:
        print("No PDF files found in data/ directory. Please add PDF files first.")
        return
    
    print(f"Loaded {len(documents)} pages from PDFs")
    
    # Split into chunks
    chunks = split_documents(documents)
    
    # Create vector store
    create_vectorstore(chunks, config.chroma_dir)
    
    print("Ingestion complete!")


if __name__ == "__main__":
    main()