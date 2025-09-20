import argparse

from data_loading import load_data

def main(args):
    documents = load_data(args.docs_dir, args.wiki_path)
    print(f"Loaded {len(documents)} documents.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs-dir', '-d', required=True, help='A local data directory')
    parser.add_argument('--wiki-path', '-w', required=True, help = 'A .csv file with metadata for the Wikipedia articles')
    args = parser.parse_args()
    main(args)