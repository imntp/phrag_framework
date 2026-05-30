""" Database store constants module. """

from enum import Enum, auto

class DatabaseType(Enum):
    """Enum for database store types."""
    CHROMA   = auto()
    FAISS    = auto()
