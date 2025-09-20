import logging

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import chromadb

logger = logging.getLogger('data_ingestion')

def create_index(nodes: list[BaseNode], model_name: str, chroma_db_path: str, persist_dir: str) -> None:
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    chroma_collection = chroma_client.get_or_create_collection("main_database")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    model = HuggingFaceEmbedding(model_name=model_name)

    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=model)

    storage_context.persist(persist_dir=persist_dir)