# ConvFinQA Vector Store

A conversational financial QA system with vector search, built using LlamaIndex and Typer CLI.

---

## Features
- **Conversational Chat**: Ask questions about financial documents with context preserved across turns.
- **Embeddings**: Supports HuggingFace and OpenAI embeddings for vector search.
- **Easy Dataset Embedding**: Embed your own dataset into a persistent vector store with a single command.
- **CLI Interface**: All operations (embedding, querying, chat) are accessible via the main CLI.
- **Example Templates**: Example usages for dataloader and query engine are included in the codebase.

---

## Installation

**Install dependencies** (requires Python 3.12+):

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in your project root with the following content (see `env_example` for reference):
   ```env
  OPENAI_API_KEY=your_openai_api_key_here
  LLM_MODEL=gpt-4o
  LLM_TEMPERATURE=0.0
  TOKENIZERS_PARALLELISM = "false"
  OPENAI_API_BASE = "https://api.openai.com/v1"
  PROMPT_ID = Your_prompt_ID_here
  PROMPT_VERSION = "1"
  LLM_MODEL = "gpt-4o"
  LLM_TEMPERATURE = "0.0"
   ```


**Note:**
- The `.env` file is used to securely store your API keys and configuration. Do not commit your real `.env` file to version control.
- An example file `env_example` is provided for reference. Copy it to `.env` and fill in your details.
---

## How to Run

All main functionality is available through the CLI in `src/main.py`.

### Start Conversational Chat (with Memory)
From the project root:

```bash
python main.py chat <RECORD_ID>
```
- Replace `<RECORD_ID>` with the desired record/document ID (see dataset for available IDs).
- Example:
  ```bash
  python main.py chat Single_JKHY/2009/page_28.pdf-3
  ```
- Type your questions, or type `exit`/`quit` to end the session.

### Embedding the Dataset

To embed (or re-embed) the dataset in the vector store, use the chat CLI and type `embed` or `load` at the prompt:
```bash
python main.py chat <RECORD_ID>
```
When prompted, type:
```
embed
```
This will embed the dataset found in the `data/` folder into the respective storage directory (HuggingFace or OpenAI, depending on configuration used in main.py).

---

## Example Templates

- Example code for dataloader and query engine usage is provided in `src/dataLoader.py` and `src/qaEngine.py`.
- You can run these scripts directly for testing.

---

## Troubleshooting
- Make sure you run example commands from the `data/` directory for examples.
- If you change embedding models or storage directories or even the dataset (in `dataLoader.py` or `main.py`), rebuild the vector store before querying.
- For more advanced usage, see comments in `dataLoader.py` and `qa_engine.py`.

---

## Quick Reference
- **Build index:** `python dataLoader.py`
- **Query index:** `python qaEngine.py`