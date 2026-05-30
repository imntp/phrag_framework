""" Main module for indexing. """

import os
import sys
from dotenv import load_dotenv

PHRAG_PATH = os.environ["PHRAG_PATH"]
sys.path.append(PHRAG_PATH)
load_dotenv()

from app.indexing.indexing_system import IndexingSystem

if __name__ == "__main__":
    indexing_system = IndexingSystem()
    indexing_system.store_data("C:\\Users\\tanph\\OneDrive\\Documents\\workspace\\rag_from_scratch\\langgraph_agent\\ATMEGA8A.pdf")
