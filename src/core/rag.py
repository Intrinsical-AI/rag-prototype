# src/core/rag.py
import logging
from sqlalchemy.exc import SQLAlchemyError
from src.core.ports import RetrieverPort, GeneratorPort
from src.db import crud
from typing import Dict 
from sqlalchemy.orm import Session
from fastapi.exceptions import HTTPException


# --- Level Module Logger ---
logger = logging.getLogger(__name__)


if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) 
    
class RagService:
    def __init__(self, retriever: RetrieverPort, generator: GeneratorPort):
        self.retriever = retriever
        self.generator = generator
        logger.info(f"RagService initialized with retriever: {type(retriever).__name__} and generator: {type(generator).__name__}")

    def ask(self, db: Session, question: str, k: int = 3) -> Dict:
        logger.info(f"Processing question: '{question}' with k={k}") # Entry logs
        try:
            ids, scores = self.retriever.retrieve(question, k=k)
            logger.debug(f"Retrieved doc_ids: {ids} with scores: {scores}")
        except Exception as e:
            logger.error(f"Error during retrieval for question '{question}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error during document retrieval.") from e
            # ids = []
            # contexts = []
            # logger.warning("Proceeding with empty context due to retrieval error.")
        
        if ids: 
            docs = crud.get_documents(db, ids)
            contexts = [d.content for d in docs]
            logger.debug(f"Contexts for generation: {contexts}")
        else:
            contexts = []
            logger.info("No documents retrieved or retrieval failed, generating answer without specific context.")


        try:
            answer: str = self.generator.generate(question, contexts)
            
            logger.info(f"Generated answer for question '{question}': '{answer[:100]}...'")
        except HTTPException as http_err: 
            # Capture HTTPException first - Re-send HTTPException to avoid FastAPI handling
            logger.warning(f"Generator raised HTTPException for question '{question}': {http_err.detail}", exc_info=True)
            raise #
        except Exception as e: 
            # Capturar other no-http exceptions
            logger.error(f"Unexpected error during generation for question '{question}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="An unexpected error occurred during answer generation.") from e
            
        try:
            crud.add_history(db, question, answer)
            logger.debug(f"Successfully saved Q&A to history for question: '{question[:50]}...'")
        except SQLAlchemyError as e_sql:
            logger.error(f"SQLAlchemyError while saving Q&A to history. Q: '{question[:50]}...'. Error: {e_sql}", exc_info=True)
            # No re-levantar si queremos que la request /ask continúe
        except Exception as e_gen:
            logger.error(f"Unexpected error while saving Q&A to history. Q: '{question[:50]}...'. Error: {e_gen}", exc_info=True)
            # No re-levantar