"""
Ollama Flow - A Python library for the Ollama API.

主要功能：
- 支援 generate, chat, embed 端點
- 結構化輸出支援
- 串流模式支援
"""

from .client import OllamaClient
from .models import GenerateRequest, ChatRequest, EmbedRequest, ChatMessage
from .schemas import StructuredOutput

__version__ = "0.1.0"
__all__ = ["OllamaClient", "GenerateRequest", "ChatRequest", "EmbedRequest", "ChatMessage", "StructuredOutput"] 