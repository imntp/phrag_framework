""" Test module for base loader. """
import os
import sys
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SOURCE_DIR, "../../../../"))

from src.rag.ingestion import IngestionType, get_loader

DATA_PATH = ""
LOADER_NAME = IngestionType.PDF_FILE.name.upper()

def load_documents():
    """Test the get_loader function."""
    loader = get_loader(LOADER_NAME)
    assert loader is not None, "Loader should not be None"
    return loader.load(DATA_PATH)

if __name__ == "__main__":
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")
