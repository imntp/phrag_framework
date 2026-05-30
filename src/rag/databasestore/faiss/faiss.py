""" FAISS database module. """

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.rag.databasestore.constants import DatabaseType
from src.rag.databasestore.base_db.base_db import BaseDB

class FaissDB(BaseDB, plugin_name=DatabaseType.FAISS.name.upper()):
    """FAISS database implementation."""

    def store_documents(self, documents:list[Document], embeddings, *args, **kwargs):
        """Store a document in the FAISS database."""

        self._vector_store = FAISS.from_documents(
            documents = documents,
            embedding = embeddings,
        )
        self._vector_store.save_local(self._save_directory)

    def load_database(self, embeddings, *args, **kwargs) -> None:
        """Load documents from the FAISS database."""
        self._vector_store = FAISS.load_local(
            embeddings = embeddings,
            folder_path = self._save_directory,
        )
