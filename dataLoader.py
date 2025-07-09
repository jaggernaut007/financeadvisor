"""
dataLoader.py
-------------
This script loads a conversational financial QA dataset, chunks it into documents, embeds them using HuggingFace or OpenAI embeddings, builds a vector index, and persists the index and vector store to disk for future retrieval.
"""

import json
import os

from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Data Chunking ---
def chunk_financial_qa(data):
    """
    Convert the financial QA data into a list of llama_index Document objects.
    Each Document contains context and example Q&A pairs from the source document.
    Args:
        data (dict): The loaded JSON data from the dataset file.
    Returns:
        list[Document]: List of Document objects ready for embedding and indexing.
    """
    docs = []
    for item in data["train"]:
        doc_id = item["id"]
        doc = item["doc"]
        dialogue = item["dialogue"]
        # Combine pre_text , post text and table context for each document
        context = doc.get("pre_text", "") + "\n" +doc.get("post_text", "") + "\n" + json.dumps(doc.get("table", {}))

        qa_text = ""
        # For each Q&A pair, create a separate Document
        for q, a in zip(dialogue["conv_questions"], dialogue["conv_answers"]):
            qa_text += f"Q: {q}\nA: {a}\n\n"
        metadata = {
            "source_id": doc_id,  # Track which document this context and Q&A came from
        }
        text = f"Context:\n{context}\n\n Example Q&A:\n{qa_text}"
        docs.append(Document(text=text, extra_info=metadata))
    return docs

# --- Index Building with HuggingFace Embeddings ---
def load_data(file_path):
    """
    Loads the dataset, chunks it into Documents, creates embeddings, builds and persists the vector store.
    Uses HuggingFace BAAI/bge-small-en-v1.5 model for embedding.
    Args:
        file_path (str): Path to the JSON dataset.
    Returns:
        VectorStoreIndex: The constructed and persisted index object.
    """
    # Open the dataset file
    with open(file_path, "r") as f:
        data = json.load(f)
    docs = chunk_financial_qa(data)

    # Use HuggingFace embedding model (384-dim)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Create a persistent vector store
    persist_dir = os.path.join(os.path.dirname(__file__), '..', 'vectors', 'storage')

    # Create a storage context with the vector store and persist directory
    vector_store = SimpleVectorStore(persist_dir=persist_dir)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build the index from documents and embeddings
    index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context, embed_model=embed_model
    )
    # Persist the index and vector store to disk
    index.storage_context.persist(persist_dir=persist_dir)
    return index

# --- Index Building with OpenAI Embeddings (if needed) ---
def load_data_OA(file_path):
    """
    Same as load_data, but builds the vector store in ./storage-openai (for OpenAI embeddings).
    Args:
        file_path (str): Path to the JSON dataset.
    Returns:
        VectorStoreIndex: The constructed and persisted index object.
    """
    # Open the dataset file
    with open(file_path, "r") as f:
        data = json.load(f)
    docs = chunk_financial_qa(data)

    # Construct the path to the storage directory
    current_dir = os.path.dirname(__file__)
    persist_dir = os.path.join(current_dir, 'vectors', 'storage-openai')

    # Create a storage context with the vector store and persist directory
    vector_store = SimpleVectorStore(persist_dir=persist_dir)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build the index from documents and embeddings
    index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context
    )
    # Persist the index and vector store to disk
    index.storage_context.persist(persist_dir=persist_dir)
    
    return index

def loadInit(embedType="huggingface"):
    """Load document using openAI or hugging face model, based on input"""

    # Construct the path to the storage directory
    current_dir = os.path.dirname(__file__)
    print(f"Current directory: {current_dir}")

    dataset = os.path.join(current_dir, 'data', 'convfinqa_dataset.json')

    # Load the index using the OpenAI or Hugging Face vector store
    if embedType == "huggingface":
        index = load_data(dataset)
    elif embedType == "openai":
        index = load_data_OA(dataset)

    return index

if __name__ == "__main__":
    
    # Main entry point: build and persist the vector store in storage
    index = loadInit("huggingface")
    print("Index: ", index)
