from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core import download_loader


def read_local_documents(data_dir: str) -> list[Document]:
    return SimpleDirectoryReader(
        input_dir=data_dir,
    ).load_data()

def read_wikipedia_article() -> list[Document]:
    WikipediaReader = download_loader("WikipediaReader")
    loader = WikipediaReader()
    docs = loader.load_data(pages=['Megaraptora'])
    return docs


def load_data(data_dir: str) -> None:
    documents = read_local_documents(data_dir)
    documents.extend(read_wikipedia_article())
    return documents

