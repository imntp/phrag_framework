""" Constants for the ingestion module. """

from enum import Enum, auto

class IngestionType(Enum):
    """ Enumeration for different types of ingestion. """
    TEXT_FILE     = auto()
    PDF_FILE      = auto()
