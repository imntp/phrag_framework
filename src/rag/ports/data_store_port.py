"""
RAG data store port module initialization.
"""
from typing import Protocol, Sequence

class DataStore(Protocol):
    """
    Interface for RAG data store port.
    """
    def store(self, data: Sequence) -> None:
        """
        Method to store the given data.

        Args:
            data: The data to be stored.
            collection_name: The name of the collection where the data will be stored.
            persist_dir: The directory where the data will be persisted.

        Returns:
            None
        """

    def load(self) -> Sequence:
        """
        Method to load the stored data.

        Args:
            The collection to be loaded is the one specified during initialization.

        Returns:
            The loaded data.
        """
