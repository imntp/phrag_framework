""" Fixed size chunking module. """

from typing import Union
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from src.rag.chunking.constants import ChunkingType
from src.rag.chunking.base_chunker.base_chunker import BaseChunker

class FixedSizeChunker(BaseChunker, plugin_name=ChunkingType.FIXED_SIZE.name.upper()):
    """ Fixed size chunking class. """

    def __init__(self, *args, **kwargs): # pylint: disable=unused-argument
        self._loader = CharacterTextSplitter(
            separator       = kwargs["separator"],
            chunk_size      = kwargs["chunk_size"],
            chunk_overlap   = kwargs["chunk_overlap"],
            length_function = len
        )

    def chunk(self, text: Union[str, list[Document]]) -> list[str]:
        """ Chunk the input text into fixed-size chunks. """
        if isinstance(text, list):
            return self._loader.split_documents(text)
        elif isinstance(text, str):
            return self._loader.create_documents([text])
        else:
            raise ValueError(f"Not implemented for type {type(text)}")
