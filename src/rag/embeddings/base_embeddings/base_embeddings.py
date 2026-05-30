""" Base embedding module. """

from abc import ABC, abstractmethod

class BaseEmbeddings(ABC):
    """Base class for all embedding implementations."""
    registry = {}

    def __init_subclass__(cls, plugin_name:str, **kwargs):
        """Ensure that subclasses implement the embed method."""
        super().__init_subclass__(**kwargs)
        BaseEmbeddings.registry[plugin_name] = cls

    def __init__(self, *args, **kwargs): # pylint: disable=unused-argument
        """Initialize the embedding module."""
        self._embeddings = None

    @property
    def embeddings(self):
        """Return the embedding instance."""
        return self._embeddings

    @abstractmethod
    def embed(self, documents):
        """Embed data into vector format."""
        raise NotImplementedError("Subclasses must implement the embed method.")
