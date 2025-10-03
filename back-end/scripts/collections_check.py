import os
import logging

import chromadb

from pathlib import Path
from dotenv import load_dotenv

def main():
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
    persistence_dir_path = os.getenv("PERSISTENCE_DIR")
    chroma_client = chromadb.PersistentClient(path=persistence_dir_path)
    collections = chroma_client.list_collections()
    print("Available collections:")
    for coll in collections:
        print(f"- {coll.name}")

if __name__ == '__main__':
    main()