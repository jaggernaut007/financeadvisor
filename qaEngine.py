# qa_engine.py: This module provides functionality to load and query a document index.
# It uses the Llama Index library to create a vector store and query engine, 
# allowing for only efficient retrieval of relevant documents based on a given query.

# Import necessary libraries
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI

from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter
from urllib3 import response
import os
from dotenv import load_dotenv

def create_chat_engine(doc_id, llm=None, token_limit=1500, system_prompt=None, embedType=None):
    """
    Create a ChatEngine with memory support for a given doc_id.
    Args:
        doc_id (str): The document ID to filter by.
        llm: Optional LLM instance. If None, uses default.
        token_limit (int): Token limit for the memory buffer.
        system_prompt (str): Optional system prompt for the chat engine.
        embedType (str): Optional embedding type for the chat engine. "openai" or "huggingface"
    Returns:
        chat_engine: The chat engine with memory.
    """
    # Load the index based on the embedding type
    
    if embedType == "huggingface":
        index = load_index()
    elif embedType == "openai":
        index = load_index_OA()
    
    # Create a retriever with the filters
    filters = MetadataFilters(filters=[MetadataFilter(key="source_id", value=doc_id)])
    retriever = index.as_retriever(filters=filters)

    # Create a memory buffer for chat history
    memory = ChatMemoryBuffer.from_defaults(token_limit=token_limit)

    # Prepare prefix_messages for ContextChatEngine
    prefix_messages = []
    if system_prompt:
            # Use prompt_id for OpenAI LLM if provided, and also add as prefix message
            if hasattr(llm, "with_config"):
                PROMPT_ID = os.getenv("PROMPT_ID")
                VERSION = os.getenv("PROMPT_VERSION")
                llm = llm.with_config(id=PROMPT_ID, version=VERSION)
            prefix_messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
            ]
    else:
        prefix_messages = []
    # Create a chat engine with memory support

    # from openai import OpenAI
    # client = OpenAI()

    # response = client.responses.create(
    # prompt={
    #     "id": "pmpt_686e481fb03881978df9352623241cc00da5d10a2189ec2d",
    #     "version": "1"
    # }
    # )
    chat_engine = ContextChatEngine(
        retriever=retriever,
        llm=llm,
        memory=memory,
        prefix_messages=prefix_messages
    )
    return chat_engine

# Load the document index from disk
def load_index():
    """
    Load the document index from disk using the Hugging Face embedding model.
    
    Returns:
        index (LlamaIndex): The loaded document index.
    """
    # Create a Hugging Face embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Load the vector store from disk
    # Use path relative to src/ directory, works if run from project root or src/
    storage_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'storage')

    # Load the vector store from disk
    vector_store = SimpleVectorStore.from_persist_dir(storage_dir)
    # Create a storage context with the vector store and persist directory
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=storage_dir
    )

    # Load the index from storage using the embedding model
    index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)
    return index


# Load the document index from disk (OpenAI version)
def load_index_OA():
    """
    Load the document index from disk using the OpenAI vector store.
    
    Returns:
        index (LlamaIndex): The loaded document index.
    """
    # Get the absolute path to the current file
    current_dir = os.path.dirname(__file__)
    # Get the parent directory of the current file (i.e., the project root)
    project_root = os.path.dirname(current_dir)
    # Construct the path to the storage directory
    persist_dir = os.path.join(project_root, 'data', 'storage-openai')

    # Load the vector store from disk
    vector_store = SimpleVectorStore.from_persist_dir(persist_dir)
    
    # Create a storage context with the vector store and persist directory
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=persist_dir
    )
    
    # Load the index from storage
    index = load_index_from_storage(storage_context=storage_context)
    
    return index


# Query the document index (Retrieval only) - Not in use with app
def query_index(query, doc_id):
    """
    Query the document index using the given query and filter by document ID.
    
    Args:
        query (str): The query to search for.
        doc_id (str): The document ID to filter by.
    
    Returns:
        response (LlamaResponse): The query response.
    """
    # Set the desired document ID
    desired_doc_id = doc_id
    
    # Load the index using the OpenAI vector store
    # Make sure the OPENAI_API_KEY environment variable is set
    
    # index = load_index_OA()  # for OpenAI vector store
    index = load_index()  # for Hugging Face vector store
    
    # Create a metadata filter to filter by document ID
    filters = MetadataFilters(
        filters=[MetadataFilter(key="source_id", value=desired_doc_id)]
    )
    
    # Create a retriever with the filters
    retriever = index.as_retriever(filters=filters)
    
    # Create a query engine with the retriever
    query_engine = RetrieverQueryEngine(retriever=retriever)
    
    # Query the index
    response = query_engine.query(query)
    
    return response


if __name__ == "__main__":
    # # Example usage: query the index
    # response = query_index("What is the net cash from operating activities in 2009?", "Single_JKHY/2009/page_28.pdf-3")
    # print("LLM Retreiver answer:", response)
    query = "What is the net cash from operating activities in 2009?"
    doc_id = "Single_JKHY/2009/page_28.pdf-3"
    llm = OpenAI(model="gpt-4-turbo", temperature=0.0)
    # Example usage: chat with the index
    chat_engine = create_chat_engine(
        doc_id,
        llm=llm,
        system_prompt="you are a financial analyst. Answer the questions based on the context provided",
        embedType="openai"
    )
    response = chat_engine.chat(query)
    print("LLM Chat answer:", response)
    