"""
Main module for the chunking functionality in the RAG framework.
"""
import os
import sys

from dotenv import load_dotenv

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, "..", ".."))
RAG_DIR = os.path.join(ROOT_DIR, "src", "rag")
sys.path.append(RAG_DIR)

load_dotenv(os.path.join(ROOT_DIR, ".env"))

from builder import build_chunking

if __name__ == "__main__":
    data_path = r""  # Replace with the actual path to your data
    chunking = build_chunking()
    chunking.chunk_and_store(data_path)
