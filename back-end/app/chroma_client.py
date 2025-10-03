# app/chroma_client.py
import chromadb

from app.config import get_config


def get_chroma_collection():
    """Initialize and return the ChromaDB collection with proper error handling."""
    config = get_config()
    try:
        # Initialize a persistent client
        chroma_client = chromadb.PersistentClient(
            path=config.chroma.persistence_path
        )
        
        # WTF DUDE????????????????????????????????
        # Get collection name from environment or use default
        collection_name = config.chroma.collection_name
        if not collection_name:
            # List available collections to help debug
            collections = chroma_client.list_collections()
            available_names = [coll.name for coll in collections]
            print(f"Available collections: {available_names}")
            
            if available_names:
                # Use the first available collection
                collection_name = available_names[0]
                print(f"Using collection: {collection_name}")
            else:
                raise ValueError("No collections found and COLLECTION_NAME not set")
        
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Successfully loaded collection: {collection_name}")
        return collection
        
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        raise

# Initialize collection
collection = get_chroma_collection()