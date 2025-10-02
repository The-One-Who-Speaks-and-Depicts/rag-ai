import os

import chromadb

def main():
    chroma_client = chromadb.PersistentClient(path=os.getenv("PERSISTENCE_DIR"))
    collections = chroma_client.list_collections()
    print("Available collections:")
    for coll in collections:
        print(f"- {coll.name}")

if __name__ == '__main__':
    main()