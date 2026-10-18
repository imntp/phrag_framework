"""
PDF loader adapter module initialization.
"""

from typing import Sequence
from langchain_community.document_loaders import PyPDFLoader

from ports.data_loader_port import DataLoader

class PDFLoader(DataLoader):
    """
    PDF loader adapter implementation.
    """
    def load_data(self, file_path: str) -> Sequence:
        """
        Method to load data from a PDF file.

        Args:
            file_path: The path to the PDF file to be loaded.

        Returns:
            The loaded data as a sequence.
        """
        # Implement PDF loading logic here
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        assert pages is not None, "Failed to load pages from PDF."
        return pages
