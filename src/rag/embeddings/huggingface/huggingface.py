""" HuggingFace embedding module. """

from langchain_community.embeddings import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings

from src.rag.embeddings.constants import EmbeddingType
from src.rag.embeddings.base_embeddings.base_embeddings import BaseEmbeddings

class HuggingFaceEmbeddings(BaseEmbeddings, plugin_name=EmbeddingType.HUGGINGFACE.name.upper()):
    """HuggingFace embedding implementation."""

    def __init__(self, *args, **kwargs): # pylint: disable=unused-argument
        """Initialize the HuggingFace embedding module."""
        super().__init__(*args, **kwargs)
        self._embeddings = LCHuggingFaceEmbeddings(model_name=kwargs.get("model_name"))

    def embed(self, documents):
        """Embed data using HuggingFace embeddings."""
        return self._embeddings.embed_documents(documents)
