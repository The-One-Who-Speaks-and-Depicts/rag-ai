import chromadb

from app.config import get_config

def main():
    config = get_config()
    persistence_dir_path = config.chroma.persistence_path
    chroma_client = chromadb.PersistentClient(path=persistence_dir_path)
    collections = chroma_client.list_collections()
    print("Available collections:")
    for coll in collections:
        print(f"- {coll.name}")

if __name__ == '__main__':
    main()