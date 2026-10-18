"""
Fixed size chunking data splitter adapter module initialization.
"""
from typing import Sequence
from langchain_text_splitters import CharacterTextSplitter

from ports.data_splitter_port import DataSplitter

class FixedSizeChunking(DataSplitter):
    """
    Fixed size chunking data splitter adapter implementation.
    """
    def __init__(self, chunk_size: int, chunk_overlap: int):

        self.splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split(self, data: Sequence) -> Sequence:
        """
        Method to split the given data into fixed size chunks.

        Args:
            data: The data to be split.

        Returns:
            The data split into fixed size chunks.
        """
        return self.splitter.split_documents(data)
