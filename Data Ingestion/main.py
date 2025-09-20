import argparse

from data_loading import load_data

def main(args):
    documents = load_data(args.data_dir)
    print(f"Loaded {len(documents)} documents.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', required=True, help='A name of environment variable with data directory')
    args = parser.parse_args()
    main(args)