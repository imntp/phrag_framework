"""
RAG data splitter port module initialization.
"""
from typing import Protocol, Sequence

class DataSplitter(Protocol):
    """
    Interface for RAG data splitter port.
    """
    def split(self, data: Sequence) -> Sequence:
        """
        Method to split the given data into multiple parts.

        Args:
            data: The data to be split.

        Returns:
            The split data.
        """
