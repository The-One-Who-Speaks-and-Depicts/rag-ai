from datetime import datetime
import pandas as pd


from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core import download_loader


def read_local_documents(data_dir: str) -> list[Document]:
    return SimpleDirectoryReader(
        input_dir=data_dir,
    ).load_data()


def read_wikipedia_article(wiki_path: str) -> list[Document]:
    WikipediaReader = download_loader("WikipediaReader")
    loader = WikipediaReader()
    article_data = pd.read_csv(wiki_path, sep=';')
    raw_articles = loader.load_data(pages=article_data['name'])
    processed_articles = []
    for idx, article in enumerate(raw_articles):
        existing_meta = article.metadata
        article.metadata = {
            **existing_meta,
            **article_data.iloc[[idx]].to_dict('records')[0],
            "ingestion_date": str(datetime.today().date()),
        }
        processed_articles.append(article)
    return processed_articles


def load_data(data_dir: str, wiki_path: str) -> list[Document]:
    documents = read_local_documents(data_dir)
    documents.extend(read_wikipedia_article(wiki_path))
    return documents

