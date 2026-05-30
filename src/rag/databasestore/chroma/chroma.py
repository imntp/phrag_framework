""" Chroma database module. """

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.rag.databasestore.constants import DatabaseType
from src.rag.databasestore.base_db.base_db import BaseDB

class ChromaDB(BaseDB, plugin_name=DatabaseType.CHROMA.name.upper()):
    """Chroma database implementation."""

    def store_documents(self, documents:list[Document], embeddings, *args, **kwargs) -> None:
        """Store a document in the Chroma database."""

        self._vector_store = Chroma.from_documents(
            documents = documents,
            embedding = embeddings,
            persist_directory = self._save_directory,
            collection_name = self._collection_name
        )

    def load_database(self, embeddings, *args, **kwargs) -> None:
        """Load documents from the Chroma database."""
        self._vector_store = Chroma(
            persist_directory = self._save_directory,
            collection_name = self._collection_name,
            embedding_function = embeddings
        )
