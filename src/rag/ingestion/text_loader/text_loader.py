""" PDF Loader module. """

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader as LCTextLoader

from src.rag.ingestion.constants import IngestionType
from src.rag.ingestion.base_loader.base_loader import BaseLoader

class TextLoader(BaseLoader, plugin_name=IngestionType.TEXT_FILE.name.upper()):
    """Loader for PDF files."""

    def load(self, path:str) -> list[Document]:
        """Load PDF data into the vector store."""
        loader = LCTextLoader(file_path=path)
        return loader.load()
