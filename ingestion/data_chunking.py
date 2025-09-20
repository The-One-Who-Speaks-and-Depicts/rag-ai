import logging

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from llama_index.core.schema import BaseNode

logger = logging.getLogger('data_ingestion')

def chunk_data(documents: list[Document], model: str) -> list[BaseNode]:

    embed_model = HuggingFaceEmbedding(model_name=model)

    node_parser = SemanticSplitterNodeParser(
        buffer_size=1, 
        breakpoint_percentile_threshold=95,
        embed_model=embed_model
    )
    
    return node_parser.get_nodes_from_documents(documents)





