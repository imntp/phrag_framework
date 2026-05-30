""" Test the chunking module. """

import os
import sys
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SOURCE_DIR, "../../../../"))

from src.rag.ingestion import IngestionType, get_loader
from src.rag.chunking import ChunkingType, get_chunker

DATA_PATH = ""
LOADER_NAME = IngestionType.PDF_FILE.name.upper()
CHUNKER_NAME = ChunkingType.FIXED_SIZE.name.upper()
SAMPLE_TEXT = """
    Machine learning has transformed how we approach data analysis.
    The field encompasses supervised learning, unsupervised learning, and reinforcement learning.
    Supervised learning uses labeled data to train models. Common algorithms include linear regression,
    decision trees, and neural networks. These models learn patterns from examples and make predictions
    on new data.
    Unsupervised learning finds patterns in unlabeled data. Clustering algorithms like K-means
    group similar data points together. Dimensionality reduction techniques like PCA help visualize
    high dimensional data.
    Reinforcement learning takes a different approach. Agents learn by interacting with an environment,
    receiving rewards or penalties for their actions. This technique powers game playing AI and
    robotics applications.
"""

def load_documents():
    """Test the get_loader function."""
    loader = get_loader(LOADER_NAME)
    assert loader is not None, "Loader should not be None"
    return loader.load(DATA_PATH)

def test_chunking(documents):
    """Test the get_chunker function."""
    chunker = get_chunker(
        CHUNKER_NAME,
        separator   = "",
        chunk_size  = 200,
        chunk_overlap = 50
    )
    assert chunker is not None, "Chunker should not be None"
    chunks = chunker.chunk(documents)
    print(f"Chunked into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    # spitted_docs = test_chunking(load_documents())
    spitted_docs = test_chunking(SAMPLE_TEXT)
    print(f"First chunk: {spitted_docs[0]}")
    print(f"First chunk length: {len(spitted_docs[0].page_content)}")
