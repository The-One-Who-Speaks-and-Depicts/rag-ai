# app/chroma_client.py
import chromadb
from chromadb.config import Settings
import os

# Initialize a persistent client
chroma_client = chromadb.PersistentClient(
    path=os.getenv("PERSISTENCE_DIR"), # Path to your persisted index directory
    settings=Settings(allow_reset=True)
)

# Get the collection you created and evaluated
collection = chroma_client.get_collection(name=os.getenv("COLLECTION_NAME")) # Use your actual collection name