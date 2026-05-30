""" Base database module. """

from abc import ABC, abstractmethod
from langchain_core.documents import Document

class BaseDB(ABC):
    """Base class for all database implementations."""
    registry = {}

    def __init_subclass__(cls, plugin_name:str, **kwargs):
        """Ensure that subclasses implement the required methods."""
        super().__init_subclass__(**kwargs)
        BaseDB.registry[plugin_name] = cls

    def __init__(self, *args, **kwargs): # pylint: disable=unused-argument
        """Initialize the database connection."""
        self._vector_store      = None
        self._save_directory    = kwargs.get("save_directory", None)
        self._collection_name   = kwargs.get("collection_name", None)

    @property
    def vector_store(self):
        """Return the vector store instance."""
        if self._vector_store is None:
            raise ValueError("Vector store not initialized.")
        return self._vector_store

    @property
    def retriever(self):
        """Return the retriever for the vector store."""
        if self._vector_store is None:
            raise ValueError("Vector store not initialized.")
        return self._vector_store.as_retriever()

    @abstractmethod
    def store_documents(self, documents:list[Document], embeddings, *args, **kwargs) -> None:
        """Store a document in the database."""
        raise NotImplementedError("Subclasses must implement the store_documents method.")

    @abstractmethod
    def load_database(self, embeddings, *args, **kwargs) -> None:
        """Load documents from the database."""
        raise NotImplementedError("Subclasses must implement the load_database method.")
