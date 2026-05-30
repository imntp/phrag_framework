""" OpenAI embedding module. """

from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings

from src.rag.embeddings.constants import EmbeddingType
from src.rag.embeddings.base_embeddings.base_embeddings import BaseEmbeddings

class OpenAIEmbeddings(BaseEmbeddings, plugin_name=EmbeddingType.OPENAI.name.upper()):
    """OpenAI embedding implementation."""

    def __init__(self, *args, **kwargs): # pylint: disable=unused-argument
        """Initialize the OpenAI embedding module."""
        super().__init__(*args, **kwargs)
        self._embeddings = LCOpenAIEmbeddings(model=kwargs.get("model_name"))

    def embed(self, documents):
        """Embed data using OpenAI embeddings."""
        return self._embeddings.embed_documents(documents)
