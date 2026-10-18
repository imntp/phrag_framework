"""
Configuration settings for the chunking module.
"""

from dataclasses import dataclass, field
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


DEFAULT_EMBED_MODEL = "text-embedding-3-small"


@dataclass
class Settings:
    """Settings for chunking """

    chunk_size: int = 1024
    overlap_size: int = 256

    embed_model: Embeddings = field(
        default_factory=lambda: OpenAIEmbeddings(model=DEFAULT_EMBED_MODEL)
    )
    collection_name: str = "default"
    persist_directory: str = "default"

settings = Settings()