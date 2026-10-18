"""
Chunking builder module for the RAG framework.
"""

from core.chunking import Chunking
from adapters.data_loader.pdf_loader import PDFLoader
from adapters.data_splitter.fixed_size_chunking import FixedSizeChunking
from adapters.data_store.chromadb_store import ChromaDBStore

from config import Settings
from config import settings as default_settings


def build_chunking(settings: Settings = default_settings):
    """Builds and returns a Chunking instance with default components."""

    data_loader = PDFLoader()
    data_splitter = FixedSizeChunking(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.overlap_size
    )
    data_store = ChromaDBStore(
        embeddings=settings.embed_model,
        collection_name=settings.collection_name,
        persist_directory=settings.persist_directory
    )

    return Chunking(
        data_loader=data_loader,
        data_splitter=data_splitter,
        data_store=data_store
    )
