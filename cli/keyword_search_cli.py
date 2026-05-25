#!/usr/bin/env python3

import argparse
import math
import ssl

import nltk
from constants import BM25_B, BM25_K1

from .keyword_search import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser = subparsers.add_parser("tf")
    search_parser.add_argument("doc_id", type=str, help="Enter document id")
    search_parser.add_argument("term", type=str, help="Enter search term")
    search_parser = subparsers.add_parser("idf")
    search_parser.add_argument("term", type=str, help="Enter search term")
    search_parser = subparsers.add_parser("tfidf")
    search_parser.add_argument("doc_id", type=str, help="Enter document id")
    search_parser.add_argument("term", type=str, help="Enter search term")
    search_parser = subparsers.add_parser("build")
    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    # Fix SSL on macOS
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("stopwords", quiet=True)

    match args.command:
        case "search":
            i = InvertedIndex()
            try:
                i.load()
                token = args.command.split()
                docs = dict()
                for token in list(i.index)[:5]:
                    docs_ids = list(i.index[token])
                    for docs_id in docs_ids:
                        docs[docs_id] = i.docmap[docs_id]
            except FileNotFoundError:
                print("File not found")
            string = ""
            for doc in docs.values():
                string += f"{doc['id']} {doc['title']}\n"
            pass
        case "build":
            i = InvertedIndex()
            i.build()
            i.save()
            pass
        case "tf":
            id = int(args.doc_id)
            term = args.term
            i = InvertedIndex()
            i.load()
            print(i.get_tf(id, term))
        case "idf":
            i = InvertedIndex()
            i.load()
            arg = args.term
            tokenize = i._InvertedIndex__tokenize(arg)
            stemmed_term = tokenize[0]
            total_doc_count = len(i.docmap)
            term_match_doc_count = 0
            is_set = i.index.get(stemmed_term, set())
            term_match_doc_count += len(list(is_set))

            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            i = InvertedIndex()
            i.load()
            arg = args.term
            id = int(args.doc_id)
            tokenize = i._InvertedIndex__tokenize(arg)
            stemmed_term = tokenize[0]

            total_doc_count = len(i.docmap)
            term_match_doc_count = 0
            is_set = i.index.get(stemmed_term, set())
            term_match_doc_count = len(list(is_set))

            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            tf = i.get_tf(id, stemmed_term)

            tf_idf = tf * idf

            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            i = InvertedIndex()
            bm25idf = i.bm25_idf_command(args.term)

            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            i = InvertedIndex()
            bm25tf = i.bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )

        case "bm25search":
            i = InvertedIndex()
            res = i.bm25_command(args.query)

            string = ""

            for item in list(list(res.items())):
                string += f"{item[0]}  - Score: {item[1]} \n"

            print(string)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
