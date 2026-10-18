"""
RAG data loader port module initialization.
"""

from typing import Protocol

class DataLoader(Protocol):
    """
    Interface for RAG data loader port.
    """
    def load_data(self, source):
        """
        Method to load data from the given source.

        Args:
            source: The source from which to load data.

        Returns:
            The loaded data.
        """
