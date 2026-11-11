# Agentic RAG PDF Assistant

A Streamlit app that lets you ask questions about uploaded PDF documents. It splits PDF text into chunks, embeds them with Google Gemini, retrieves relevant passages, and uses a Gemini chat model to answer questions.

## Requirements

- Python 3.10 or later
- A Google AI API key with access to the Gemini models used by the app

## Setup

From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the app dependencies:

```bash
python -m pip install streamlit python-dotenv pypdf langchain langchain-community langchain-text-splitters langchain-google-genai langgraph
```

Create a `.env` file in the repository root and add your API key:

```dotenv
GOOGLE_API_KEY=your_google_ai_api_key
```

## Run

Run Streamlit from the `apps` directory so uploaded PDFs are saved in `apps/doc_files`:

```bash
cd apps
mkdir -p doc_files
streamlit run main.py
```

Open the local URL printed by Streamlit. Upload one or more PDF files, wait for processing to finish, then ask questions in the chat.

## How it works

1. PDFs in the upload are loaded and split into overlapping text chunks.
2. The chunks are embedded and stored in an in-memory vector store.
3. For each question, an agent retrieves relevant chunks and generates an answer using Gemini.

The vector store and chat state are held in memory, so they are not persisted across app restarts. Uploaded PDF files are written to `apps/doc_files`.
