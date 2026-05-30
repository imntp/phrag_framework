""" Indexing system module. """

import os
import configparser
from src.rag.rag_builder.rag_builder import RAGBuilder

PHRAG_PATH = os.environ["PHRAG_PATH"]
config = configparser.ConfigParser()
config.read(os.path.join(PHRAG_PATH, "app/indexing/config.ini"))

BaseRAGSystem = RAGBuilder()\
    .build_loader(
        plugin_name = config["loader"]["plugin_name"]
    )\
    .build_chunker(
        plugin_name   = config["chunker"]["plugin_name"],
        chunk_size    = int(config["chunker"]["chunk_size"]),
        chunk_overlap = int(config["chunker"]["chunk_overlap"]),
    )\
    .build_embeddings(
        plugin_name = config["embeddings"]["plugin_name"],
        model_name  = config["embeddings"]["model_name"]
    )\
    .build_database(
        plugin_name = config["vectorstore"]["plugin_name"],
        save_directory = config["vectorstore"]["save_directory"],
        collection_name = config["vectorstore"]["collection_name"]
    )\
    .build()

class IndexingSystem(BaseRAGSystem):
    """ Indexing system that inherits from the built RAG system. """

    def store_data(self, path: str):
        """ Store data in the vector store. """
        documents = self.loader.load(path)
        print(f"Loaded {len(documents)} documents.")
        splitted = self.chunker.chunk(documents)
        print(f"Chunked into {len(splitted)} chunks.")
        self.database.store_documents(
            documents = splitted,
            embeddings = self.embeddings.embeddings
        )
