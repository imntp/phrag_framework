""" Builder module. """

from src.rag.ingestion import BaseLoader
from src.rag.chunking import BaseChunker
from src.rag.embeddings import BaseEmbeddings
from src.rag.databasestore import BaseDB

class RAGBuilder:
    """Builder class for constructing various components of the RAG system."""

    def __init__(self):
        """Initialize the builder."""
        self._attributes = {}

    def build_loader(self, plugin_name:str, *args, **kwargs):
        """Build a loader instance based on the plugin name."""
        if plugin_name not in BaseLoader.registry:
            raise ValueError(f"Loader plugin '{plugin_name}' not found.")
        self._attributes['loader'] = BaseLoader.registry[plugin_name](*args, **kwargs)
        return self

    def build_chunker(self, plugin_name:str, *args, **kwargs):
        """Build a chunker instance based on the plugin name."""
        if plugin_name not in BaseChunker.registry:
            raise ValueError(f"Chunker plugin '{plugin_name}' not found.")
        self._attributes['chunker'] = BaseChunker.registry[plugin_name](*args, **kwargs)
        return self

    def build_embeddings(self, plugin_name:str, *args, **kwargs):
        """Build an embeddings instance based on the plugin name."""
        if plugin_name not in BaseEmbeddings.registry:
            raise ValueError(f"Embeddings plugin '{plugin_name}' not found.")
        self._attributes['embeddings'] = BaseEmbeddings.registry[plugin_name](*args, **kwargs)
        return self

    def build_database(self, plugin_name:str, *args, **kwargs):
        """Build a database instance based on the plugin name."""
        if plugin_name not in BaseDB.registry:
            raise ValueError(f"Database plugin '{plugin_name}' not found.")
        self._attributes['database'] = BaseDB.registry[plugin_name](*args, **kwargs)
        return self


    def build(self):
        """Build the RAG system."""
        class BaseRAGSystem:
            """ RAG System class that holds all components. """
            loader = None
            chunker = None
            embeddings = None
            database = None

        # Assign the built components to the BaseRAGSystem class
        for key, value in self._attributes.items():
            setattr(BaseRAGSystem, key, value)

        return BaseRAGSystem
