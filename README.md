# AI-Q-A-API
A FastAPI based RAG service that answers questions from a PDF using FAISS, SentenceTransformers, and Ollama with structured logging and endpoint.

# RAG PDF Q&A Service

A FastAPI-based Retrieval-Augmented Generation (RAG) application that answers questions from a local PDF document. The app extracts text from a PDF, splits it into searchable chunks, builds a FAISS vector index using SentenceTransformers embeddings, retrieves the most relevant chunks for a user question, and sends the retrieved context to a local Ollama model for answer generation.

## Features

- Extracts text from a PDF knowledge base
- Splits PDF content into searchable chunks
- Generates embeddings with `sentence-transformers`
- Stores and searches embeddings using FAISS
- Uses Ollama for local LLM response generation
- Provides a FastAPI REST endpoint for asking questions
- Includes request logging and pipeline logs
- Supports Swagger UI for testing the API

## Project Structure

```text
AI_1/
|-- rag_fastapi.py              # FastAPI RAG API service
|-- Knowledge_base_for_RAG.pdf  # Source PDF knowledge base
|-- req.txt                     # Python dependencies
|-- pdf_chunks.json             # Generated/stored PDF chunks
|-- process_logs.txt            # Runtime logs
|-- rag.py                      # RAG experiment script
|-- rag1.py                     # RAG experiment script
|-- rag3.py                     # RAG experiment script
|-- rag4.py                     # RAG experiment script
|-- pdf.py                      # PDF processing script
`-- main.py                     # Additional project script
```

## Requirements

- Python 3.10 or newer
- Ollama installed and running locally
- The Ollama model used by the app pulled locally

The default model in `rag_fastapi.py` is:

```python
OLLAMA_MODEL = "llama3.2"
```

Pull the model with:

```bash
ollama pull llama3.2
```

## Installation

1. Clone the repository:

```bash
git clone <your-repository-url>
cd AI_1
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r req.txt
```

4. Make sure Ollama is running:

```bash
ollama serve
```

In another terminal, make sure the model is available:

```bash
ollama pull llama3.2
```

## Configuration

Update these values in `rag_fastapi.py` if needed:

```python
PDF_PATH = r"C:\Users\admin\PycharmProjects\AI_1\Knowledge_base_for_RAG.pdf"
OLLAMA_MODEL = "llama3.2"
TOP_K = 3
```

Recommended change before sharing the project on GitHub:

```python
PDF_PATH = "Knowledge_base_for_RAG.pdf"
```

This makes the app portable across different machines.

## Running the API

Start the FastAPI server:

```bash
python rag_fastapi.py
```

Or run it with Uvicorn:

```bash
uvicorn rag_fastapi:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

Swagger UI is available at:

```text
http://localhost:8000/docs
```

## API Usage

### Ask a Question

Endpoint:

```http
POST /ask
```

Request body:

```json
{
  "question": "What is this document about?"
}
```

Example response:

```json
{
  "answer": "The answer generated from the retrieved PDF context."
}
```

Example using `curl`:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is this document about?\"}"
```

## How It Works

1. The app loads the PDF using `pypdf`.
2. Text is split into chunks based on numbered question patterns.
3. Each chunk is converted into an embedding using `all-MiniLM-L6-v2`.
4. FAISS builds an in-memory vector index from the chunk embeddings.
5. When a user asks a question, the app embeds the question and retrieves the top matching chunks.
6. Retrieved chunks are passed as context to Ollama.
7. Ollama generates an answer based only on the retrieved context.

## Logging

The application writes logs to:

```text
process_logs.txt
```

Logs include:

- PDF loading status
- Chunk creation details
- FAISS index build status
- Retrieved chunks and scores
- Ollama response timing
- API request and error details

## Notes

- The RAG pipeline is built once during FastAPI startup and reused for every request.
- The FAISS index is stored in memory, so it is rebuilt each time the server starts.
- The app depends on a local Ollama server, so answers will not work unless Ollama is running.
- The current implementation is designed for a single local PDF knowledge base.

## Future Improvements

- Make the PDF path configurable through environment variables
- Save and reload the FAISS index from disk
- Add file upload support for new PDFs
- Add support for multiple documents
- Add Docker support
- Add automated tests
- Improve chunking for non-FAQ-style documents

## License

This project is for learning and experimentation. Add a license file if you plan to publish or distribute it.

