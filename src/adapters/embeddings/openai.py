# src/adapters/embeddings/openai.py
from typing import List
from openai import OpenAI, APIError

from src.core.ports import EmbedderPort 
from src.settings import settings

import logging
logger = logging.getLogger(__name__)

class OpenAIEmbedder(EmbedderPort): # DI
    DIM: int = 1536 
    
    def __init__(self):
        # API key from settings.openai_api_key or env variable
        self.client = OpenAI(
            api_key=settings.openai_api_key
        )
        if settings.openai_embedding_model == "text-embedding-3-large":
            self.DIM = 3072
        elif settings.openai_embedding_model == "text-embedding-ada-002":
            self.DIM = 1536

    def embed(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model=settings.openai_embedding_model,
                input=text
                # 'encoding_format': 'float' # default
                # 'dimensions': 1536 # 3rd gen models allows reduced dimensions
            )
            if response.data and response.data[0].embedding:
                return response.data[0].embedding
            else:
                # logger.error("OpenAI embedding response malformed.")
                # raise HTTPException(500, detail="OpenAI embedding response malformed")
                raise ValueError("OpenAI embedding response malformed: No embedding data found.")
        except APIError as err:
            logger.error(f"OpenAI API Error during embedding for text (first 50 chars): '{text[:50]}...'. Error: {err}", exc_info=True)
            raise # Re-levanta la APIError original con su traceback
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI embedding for text (first 50 chars): '{text[:50]}...'. Error: {e}", exc_info=True)
            raise # Re-levanta la excepción original