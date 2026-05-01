from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
from pypdf import PdfReader
import re
import warnings
warnings.filterwarnings("ignore")
# import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
# from fastapi.responses import JSONResponse


PDF_PATH     = r"C:\Users\admin\PycharmProjects\AI_1\Knowledge_base_for_RAG.pdf"
OLLAMA_MODEL = "llama3.2"
TOP_K        = 3    # chunks to retrieve
# CHUNKS_JSON_PATH = "pdf_chunks.json"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    handlers=[
        logging.FileHandler("process_logs.txt"),
        logging.StreamHandler()          # print to console
    ]
)

logger          = logging.getLogger("rag_app")
pipeline_logger = logging.getLogger("rag_app.pipeline")
api_logger      = logging.getLogger("rag_app.api")


class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str


def load_pdf(path: str) -> str: #extracting text from pdf
    pipeline_logger.info(f"Loading PDF from path: {path}")
    reader    = PdfReader(path)
    full_text = ""
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            full_text += text + "\n"
            pipeline_logger.debug(f"Extracted text from page {page_num} ({len(text)} chars)")
        else:
            pipeline_logger.warning(f"No text found on page {page_num}")
    pipeline_logger.info(f"PDF loading complete. Total characters extracted: {len(full_text)}")
    return full_text

def split_into_chunks(text: str) -> list[str]:#Split into chunks
    """
    Split PDF into FAQ chunks based on numbered questions.
    """
    pipeline_logger.info("Splitting text into chunks...")

    pattern       = r'(?=\d+\.\s+)'
    chunks        = re.split(pattern, text)
    cleaned_chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]

    pipeline_logger.info(f"Chunking complete. Total chunks created: {len(cleaned_chunks)}")
    for i, chunk in enumerate(cleaned_chunks):
        pipeline_logger.debug(f"Chunk {i}: {chunk[:80]}{'...' if len(chunk) > 80 else ''}")
    return cleaned_chunks


def build_index(chunks: list[str], model: SentenceTransformer):#Embed chunks & build FAISS index
    pipeline_logger.info(f"Building FAISS index for {len(chunks)} chunks...")
    t0         = time.time()
    embeddings = model.encode(chunks, convert_to_numpy=True).astype(np.float32)
    dimension  = embeddings.shape[1]
    pipeline_logger.info(f"Embeddings shape: {embeddings.shape}, dimension: {dimension}")

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    elapsed = round(time.time() - t0, 3)
    pipeline_logger.info(f"FAISS index built successfully in {elapsed}s. Total vectors: {index.ntotal}")
    return index


def retrieve(question: str, model: SentenceTransformer,
             index: faiss.IndexFlatL2, chunks: list[str], top_k: int): # Retrieve top-K chunks WITH scores
    pipeline_logger.info(f"Retrieving top-{top_k} chunks for question: '{question}'")
    t0    = time.time()
    q_emb = model.encode([question], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(q_emb, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if idx < len(chunks):
            score = round(float(1 / (1 + dist)), 4)
            results.append({
                "rank"   : rank,
                "score"  : score,
                "l2_dist": round(float(dist), 4),
                "chunk"  : chunks[idx],
            })
            pipeline_logger.info(
                f"  Rank {rank} | Score: {score} | L2: {round(float(dist), 4)} | "
                f"Preview: {chunks[idx][:80]}{'...' if len(chunks[idx]) > 80 else ''}"
            )

    elapsed = round(time.time() - t0, 3)
    pipeline_logger.info(f"Retrieval complete in {elapsed}s. {len(results)} chunks returned.")
    return results

def ask_ollama(question: str, context: str) -> str: # Ask Ollama
    pipeline_logger.info(f"Sending question to Ollama model '{OLLAMA_MODEL}'...")
    pipeline_logger.debug(f"Context length: {len(context)} chars")

    prompt = (
        "You are a helpful assistant. "
        "Answer the question using ONLY the context provided below. "
        "Do NOT say you don't have information if the answer is in the context. "
        "Quote or summarise directly from the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (based strictly on the context above):"
    )

    t0       = time.time()
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    elapsed = round(time.time() - t0, 3)
    answer  = response["message"]["content"].strip()

    pipeline_logger.info(f"Ollama responded in {elapsed}s. Answer length: {len(answer)} chars")
    pipeline_logger.debug(f"Answer preview: {answer[:120]}{'...' if len(answer) > 120 else ''}")
    return answer

class RAGPipeline: # RAG PIPELINE – built ONCE, reused every query
    def __init__(self):
        pipeline_logger.info("Initialising RAGPipeline...")
        t_start = time.time()

        pipeline_logger.info(f"Reading PDF: {PDF_PATH}")
        raw_text = load_pdf(PDF_PATH)

        self.chunks = split_into_chunks(raw_text)

        pipeline_logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        pipeline_logger.info("SentenceTransformer model loaded successfully.")

        self.index = build_index(self.chunks, self.embed_model)

        elapsed = round(time.time() - t_start, 3)
        pipeline_logger.info(f"RAGPipeline initialisation complete in {elapsed}s.")

    def ask(self, request: QuestionRequest) -> AnswerResponse:
        req_id = str(uuid.uuid4())[:8]
        pipeline_logger.info(f"[{req_id}] New question received: '{request.question}'")
        t_start = time.time()

        results = retrieve(request.question, self.embed_model,
                           self.index, self.chunks, TOP_K)

        pipeline_logger.info(f"[{req_id}] RETRIEVED CHUNKS:")
        for r in results:
            pipeline_logger.info(f"[{req_id}]   Rank {r['rank']} | Score {r['score']} | {r['chunk'][:100]}")

        context = "\n\n".join(r["chunk"] for r in results)
        answer  = ask_ollama(request.question, context)

        elapsed = round(time.time() - t_start, 3)
        pipeline_logger.info(f"[{req_id}] ask() completed in {elapsed}s.")
        return AnswerResponse(answer=answer)

pipeline: RAGPipeline | None = None # FASTAPI APP

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the RAG pipeline once at startup."""
    global pipeline
    api_logger.info("FastAPI startup: building RAG pipeline...")
    try:
        pipeline = RAGPipeline()
        api_logger.info("RAG pipeline ready. Server accepting requests.")
    except Exception as e:
        api_logger.critical(f"Failed to initialise RAG pipeline: {e}", exc_info=True)
        raise
    yield
    api_logger.info("FastAPI shutdown: cleaning up.")

app = FastAPI(
    title="RAG PDF Q&A Service",
    description="Answer questions from a PDF using a local RAG pipeline + Ollama.",
    version="1.0.0",
    lifespan=lifespan,
)

# Request/response timing middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    req_id  = str(uuid.uuid4())[:8]
    t_start = time.time()
    api_logger.info(f"[{req_id}] --> {request.method} {request.url.path}")
    response = await call_next(request)
    elapsed  = round((time.time() - t_start) * 1000, 2)
    api_logger.info(
        f"[{req_id}] <-- {request.method} {request.url.path} "
        f"| status={response.status_code} | {elapsed}ms"
    )
    return response

# Health check
@app.get("/health", summary="Health check")
async def health():
    api_logger.info("Health check endpoint called.")
    return {"status": "ok", "model": OLLAMA_MODEL, "chunks_loaded": len(pipeline.chunks) if pipeline else 0}

# Main endpoint
@app.post("/ask", response_model=AnswerResponse, summary="Ask a question")
async def ask_question(request: QuestionRequest):
    """
    POST /ask  –  Ask a question; the service retrieves relevant
    chunks from the PDF and returns an LLM-generated answer.
    """
    api_logger.info(f"POST /ask  question='{request.question}'")

    if not request.question.strip():
        api_logger.warning("Empty question received.")
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if pipeline is None:
        api_logger.error("Pipeline not initialised when /ask was called.")
        raise HTTPException(status_code=503, detail="RAG pipeline not ready.")

    try:
        response = pipeline.ask(request)
        api_logger.info(f"POST /ask  answer returned ({len(response.answer)} chars)")
        return response
    except Exception as e:
        api_logger.error(f"Error while processing question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__": # CLI ENTRYPOINT (kept for backward compat)
    import uvicorn
    logger.info("Starting RAG FastAPI server via __main__...")
    uvicorn.run("rag_fastapi:app", host="0.0.0.0", port=8000, reload=False)
