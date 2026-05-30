""" Base class for all loaders. """

from abc import ABC, abstractmethod
from langchain_core.documents import Document

class BaseLoader(ABC):
    """Base class for all loaders."""
    registry = {}

    def __init_subclass__(cls, plugin_name:str, **kwargs):
        """Ensure that subclasses implement the load method."""
        super().__init_subclass__(**kwargs)
        BaseLoader.registry[plugin_name] = cls

    @abstractmethod
    def load(self, path:str) -> list[Document]:
        """Load data into the vector store."""

def get_loader(plugin_name:str) -> BaseLoader:
    """Get the loader class for the given plugin name."""
    loader_cls = BaseLoader.registry.get(plugin_name)
    if not loader_cls:
        raise ValueError(f"No loader found for plugin name: {plugin_name}")
    return loader_cls()
