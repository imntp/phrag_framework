""" Base module for chunking. """

from typing import Union
from abc import ABC, abstractmethod

from langchain_core.documents import Document

class BaseChunker(ABC):
    """Base class for all chunking strategies."""
    registry = {}

    def __init_subclass__(cls, plugin_name:str, **kwargs):
        """Ensure that subclasses implement the chunk method."""
        super().__init_subclass__(**kwargs)
        BaseChunker.registry[plugin_name] = cls

    @abstractmethod
    def chunk(self, text: Union[str, list[Document]]) -> list[str]:
        """Chunk data into smaller pieces."""

def get_chunker(plugin_name: str, *args, **kwargs) -> BaseChunker:
    """Factory method to load a chunker by name."""
    if plugin_name not in BaseChunker.registry:
        raise ValueError(f"Chunker '{plugin_name}' not found in registry.")
    return BaseChunker.registry[plugin_name](*args, **kwargs)
