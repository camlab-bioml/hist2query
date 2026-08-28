import argparse

from hist2query.cli import (
    serve,
    train,
    add)

from hist2query._version import __version__

def main():

    parser = argparse.ArgumentParser(
        prog="hist2query")

    parser.add_argument('-v', "--version", action="version",
                        help="Show the current hist2query version then exit. Does not execute the application.",
                        version=f"This is hist2query: v{__version__}")

    subparsers = parser.add_subparsers(
        dest="command",
        required=True)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Run the FastAPI server")

    serve.configure_parser(serve_parser)

    train_parser = subparsers.add_parser(
        "train",
        help="Build and train an initial FAISS index.")

    train.configure_parser(train_parser)

    add_parser = subparsers.add_parser(
        "add",
        help="Add embeddings to a trained index.")
    add.configure_parser(add_parser)

    args = parser.parse_args()

    args.func(args)

if __name__ == "__main__":
    main() # pragma: no cover