import argparse
import logging

from data_loading import load_data
from data_chunking import chunk_data

logger = logging.getLogger('data_loading')
logger.setLevel(logging.INFO)


def main(args):
    documents = load_data(args.docs_dir, args.wiki_path)
    logger.info("Loaded %s documents.", len(documents))
    nodes = chunk_data(documents, args.hugging_face_model)
    logger.info("Split %s documents into %s nodes.", len(documents), len(nodes))
    print(f"First node text: {nodes[0].text[:200]}...")
    print(f"First node metadata: {nodes[0].metadata}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs-dir', '-d', required=True, help='A local data directory')
    parser.add_argument('--wiki-path', '-w', required=True, help = 'A .csv file with metadata for the Wikipedia articles')
    parser.add_argument('--hugging-face-model', '-hfm', help = 'An embedding model for data chunking from HuggingFace', default="BAAI/bge-small-en-v1.5")
    args = parser.parse_args()
    main(args)