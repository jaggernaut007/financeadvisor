"""
Main typer app for ConvFinQA
"""

import typer
import sys
import os
from rich import print as rich_print

# Ensure src/ is on sys.path so qa_engine can be imported when running as a script
sys.path.insert(0, os.path.dirname(__file__))
import qaEngine
import dataLoader
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

app = typer.Typer(
    name="main",
    help="CLI chat for ConvFinQA",
    add_completion=True,
    no_args_is_help=True,
)


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
) -> None:
    """Ask questions about a specific record"""
    history = []

    # Embedding type to use
    embedType = "huggingface" # "openai" or "huggingface"

    # LLM to use for RAG generation
    # Make sure the OPENAI_API_KEY environment variable is set
    MODEL = os.getenv("LLM_MODEL", "gpt-4o")
    TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Initialize the LLM with the specified model and temperature
    llm = OpenAI(model=MODEL, temperature=TEMPERATURE, api_key=OPENAI_API_KEY)

    # System prompt for the chat engine
    system_prompt="you are a financial analyst. Be concise and to the point and do not guess or speculate any answers. Answer the questions based on the context provided and do not include any additional information"
    
    # Create a chat engine with memory for this session
    chat_engine = qaEngine.create_chat_engine(
        doc_id=record_id,
        embedType=embedType,
        system_prompt=system_prompt,
        llm=llm,
    )
    
    print("Info: Type 'exit' or 'quit' to end the chat.")
    print("Info: Type 'embed' or 'load' to embed the dataset.")
    while True:
        message = input(">>> ")

        if message.strip().lower() in {"exit", "quit"}:
            break
        # Embed the dataset
        if message.strip().lower() in {"embed","load"}:
            rich_print(f"[blue][bold]assistant:[/bold] Loading dataset...[/blue]")
            dataLoader.loadInit(embedType=embedType)
            rich_print(f"[blue][bold]assistant:[/bold] Dataset loaded successfully[/blue]")
            break
        # Chat with the chat engine
        response = chat_engine.chat(message)
        rich_print(f"[blue][bold]assistant:[/bold] {response}[/blue]")
        history.append({"user": message, "assistant": response})


@app.command()
def myfunc(record_id: str,query: str) -> str:
    """My hello world function"""
    # response from Query Engine
    
    rich_print("Hello World")
    return "Hello World"



if __name__ == "__main__":
    app()
