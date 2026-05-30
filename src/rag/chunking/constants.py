""" Constants for chunking module. """

from enum import Enum, auto

class ChunkingType(Enum):
    """ Enumeration for different types of chunking. """
    FIXED_SIZE          = auto()
    TOKEN_BASED         = auto()
    RECURSIVE_CHARACTER = auto()
    SEMANTIC_BASED      = auto()
    SENTENCE_BASED      = auto()
    HIERARCHICAL_MD     = auto()
    AGNETIC             = auto()
