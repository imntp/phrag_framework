"""
Core chunking module for the RAG framework.
"""

from chunking_container import ChunkingContainer

class Chunking(ChunkingContainer):
    """
    Core chunking class for the RAG framework.
    Inherits from ChunkingContainer.
    """

    def chunk_and_store(self, data):
        """
        Chunks the given data using the data splitter and stores the chunks using the data store.

        Args:
            data: The data to be chunked and stored.
        """
        pages = self.data_loader.load_data(data)
        print(f"Loaded {len(pages)} pages from the data.")

        chunks = self.data_splitter.split(pages)
        print(f"Generated {len(chunks)} chunks from the data.")

        self.data_store.store(chunks)
        print(f"Stored {len(chunks)} chunks in the data store.")
