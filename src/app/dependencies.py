"""
File: src/app/dependencies.py
Path: src/app/dependencies.py
Dependency injection providers for the FastAPI application.
"""

from src.core.rag import RagService
from src.settings import settings
from src.adapters.embeddings.sentence_transformers import SentenceTransformerEmbedder
from src.adapters.retrieval.sparse_bm25 import SparseBM25Retriever
from src.adapters.retrieval.dense_faiss import DenseFaissRetriever
from src.adapters.generation.openai_chat import OpenAIGenerator
from src.adapters.generation.ollama_chat import OllamaGenerator
from src.core.ports import GeneratorPort
from src.db.base import SessionLocal
import requests
import logging

# ---------- RagService ----------
_rag_service: RagService | None = None

def _choose_generator() -> GeneratorPort:
    ollama_is_usable = False
    if settings.ollama_enabled:
        try:
             # Lista models as health check
            response = requests.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=2)
            if response.status_code == 200:
                logging.info("Ollama server detected and explicitly enabled. Using Ollama generator.")
                ollama_is_usable = True
                return OllamaGenerator()
            else:
                logging.warning(f"OLLAMA_ENABLED=true but could not confirm Ollama server status (tags endpoint status: {response.status_code}).")
        except requests.exceptions.RequestException as e:
            logging.warning(f"OLLAMA_ENABLED=true but server not reachable at {settings.ollama_base_url}: {e}")
    
    if settings.openai_api_key:
        logging.info("OpenAI API key found. Using OpenAI generator.")
        return OpenAIGenerator()
    
    # if not OpenAI API key and Ollama it's disabled 
    # Opción A: Try force using Ollama 
    if not ollama_is_usable:
        logging.info("OpenAI API key not found. Attempting to use Ollama as a fallback...")
        try:
            response = requests.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=2)
            if response.status_code == 200:
                logging.info("Ollama server detected and usable as fallback. Using Ollama generator.")
                return OllamaGenerator()
            else:
                logging.warning(f"Fallback to Ollama failed: Could not confirm Ollama server status (tags endpoint status: {response.status_code}).")
        except requests.exceptions.RequestException as e:
            logging.warning(f"Fallback to Ollama failed: Server not reachable at {settings.ollama_base_url}: {e}")

    # Everything failed - Error msg + raise error
    error_msg = (
        "LLM Generator could not be initialized. "
        "Please set OPENAI_API_KEY in your environment/`.env` file, "
        "or ensure Ollama is running and `ollama_enabled=True` (or accessible as fallback)."
    )
    logging.error(error_msg)
    # stop app init good if no generator available.
    raise RuntimeError(error_msg) 

def init_rag_service():
    global _rag_service

    # --- Problems with initial empty DB ---
    from src.db.crud import add_documents as crud_add_documents
    from src.db.models import Document as DbDocumentModel
    from src.utils import preprocess_text
    import csv
    from pathlib import Path

    with SessionLocal() as db_check:
        doc_count = db_check.query(DbDocumentModel).count()
        if doc_count == 0 and Path(settings.faq_csv).is_file():
            logging.info(f"Database is empty. Populating from {settings.faq_csv}...")
            texts_to_add = []
            try:
                with open(settings.faq_csv, newline="", encoding="utf-8") as fh:
                    reader = csv.reader(fh)
                    if settings.csv_has_header: 
                        next(reader, None)
                    for row in reader:
                        if row: 
                            content = f"{row[0]} {row[1]}" # Q + A                            
                            texts_to_add.append(preprocess_text(content))
                if texts_to_add:
                    crud_add_documents(db_check, texts_to_add)
                    logging.info(f"Populated database with {len(texts_to_add)} documents from CSV.")
                else:
                    logging.warning("faq.csv was empty or could not be read properly.")
            except Exception as e:
                logging.error(f"Failed to populate database from CSV: {e}", exc_info=True) 
                
    # 1) Prepare Embedder
    embedder = SentenceTransformerEmbedder()
    # 2) Retriever (based on settings)
    with SessionLocal() as db: # new sesion
        docs_orm = db.query(DbDocumentModel).all()
        contents = [d.content for d in docs_orm]
        ids = [d.id for d in docs_orm]

    if settings.retrieval_mode == "dense":
        # ASSUMING  build_index.py have been already EXECUTED!!
        if not Path(settings.index_path).is_file() or not Path(settings.id_map_path).is_file():
            logging.warning(f"Dense retrieval mode selected, but FAISS index ({settings.index_path}) or id_map ({settings.id_map_path}) not found.")
            logging.warning("Please run 'python scripts/build_index.py' with dense mode enabled.")
            logging.warning("Falling back to sparse retrieval for this session.")
            retriever = SparseBM25Retriever(contents, ids) # Fallback
        else:
            retriever = DenseFaissRetriever(embedder)

    else: # sparse
        retriever = SparseBM25Retriever(contents, ids)

    generator = _choose_generator()
    _rag_service = RagService(retriever, generator)

def get_rag_service() -> RagService:
    assert _rag_service, "RagService not initialised"
    return _rag_service