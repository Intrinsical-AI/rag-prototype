"""
File: src/adapters/generation/__init__.py
Path: src/adapters/generation/__init__.py
Generation adapters for LLM text generation.
"""

from .ollama_chat import OllamaGenerator
from .openai_chat import OpenAIGenerator

__all__ = [
    "OllamaGenerator",
    "OpenAIGenerator"
] 