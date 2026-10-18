"""
ChromaDB data store adapter module initialization.
"""

from typing import Sequence
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from ports.data_store_port import DataStore

class ChromaDBStore(DataStore):
    """
    ChromaDB data store adapter implementation.
    """

    def __init__(
            self,
            embeddings: Embeddings,
            collection_name: str,
            persist_directory: str
        ):
        if not hasattr(embeddings, "embed_documents"):
            raise TypeError(
                "embeddings must provide embed_documents(); pass an embeddings instance, not a model name."
            )
        self._embeddings = embeddings
        self._collection_name = collection_name
        self._persist_directory = persist_directory

    def store(
            self,
            data: Sequence,
        ):
        """
        Method to store the given data in ChromaDB.

        Args:
            data: The data to be stored.
        """
        Chroma.from_documents(
            documents=data,
            embedding=self._embeddings,
            persist_directory=self._persist_directory,
            collection_name=self._collection_name
        )

    def load(
            self,
        ):
        """
        Method to load the given collection from ChromaDB.

        Args:
            The collection to be loaded is the one specified during initialization.
        """
        return Chroma(
            embedding_function=self._embeddings,
            persist_directory=self._persist_directory,
            collection_name=self._collection_name
        )
