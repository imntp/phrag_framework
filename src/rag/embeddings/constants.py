""" Embedding constants module. """

from enum import Enum, auto

class EmbeddingType(str, Enum):
    """Enum for embedding types."""
    OPENAI      = auto()
    HUGGINGFACE = auto()
