"""
Chunking container module for the RAG framework.
"""

from dataclasses import dataclass
from ports.data_loader_port import DataLoader
from ports.data_splitter_port import DataSplitter
from ports.data_store_port import DataStore

@dataclass
class ChunkingContainer:
    """
    Data class representing the chunking container for the RAG framework.
    """

    data_loader: DataLoader
    data_splitter: DataSplitter
    data_store: DataStore
