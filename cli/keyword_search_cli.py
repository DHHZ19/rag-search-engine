#!/usr/bin/env python3

import argparse
import json
import math
import os
import pickle
import ssl
from collections import Counter
from nltk.stem import PorterStemmer
from constants import BM25_K1
from constants import BM25_B

import nltk


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

    class InvertedIndex:
        def __init__(self):
            self.index = {}
            self.docmap = {}
            self.term_frequencies = {}
            self.doc_lengths = {}
            self.stemmer = PorterStemmer()

        def __tokenize(self, text):
            search_term = ""
            for char in text:
                if char.isalnum() or char.isspace():
                    search_term += char
                else:
                    search_term += ""

            tokens = search_term.split()
            with open("stopwords.txt", "r") as f:
                stopwords = f.read().splitlines()
                tokens = [token for token in tokens if token.lower() not in stopwords]
            return [self.stemmer.stem(token.lower()) for token in tokens]

        def __add_document(self, doc_id, text):
            tokenize = self.__tokenize(text)
            self.doc_lengths[doc_id] = len(tokenize)

            for token in tokenize:
                if token not in self.index:
                    self.index[token] = set()
                self.index[token].add(doc_id)
                if doc_id not in self.term_frequencies:
                    self.term_frequencies[doc_id] = Counter()
                self.term_frequencies[doc_id][token] += 1

        def __get_avg_doc_length(self):
            docs_lengths = []
            docs = len(list(self.doc_lengths))

            if docs == 0:
                raise ValueError("no docs")

            for doc_length in self.doc_lengths.values():
                docs_lengths.append(doc_length)

            return sum(docs_lengths) / docs

        def get_documents(self, term):
            tokenize = self.__tokenize(term)
            if len(tokenize) > 1:
                raise ValueError("Expected a single token, got multiple")
            if len(tokenize) == 0:
                return []
            ids = sorted(self.index.get(tokenize[0], set()))
            return ids

        def build(self):
            with open("./data/movies.json", "r") as file:
                data = json.load(file)

            for key, value in data.items():
                for i, m in enumerate(value, start=1):
                    self.__add_document(i, f"{m['title']} {m['description']}")
                    if i not in self.docmap:
                        self.docmap[i] = m

        def save(self):
            if not os.path.isdir("cache"):
                os.mkdir("cache")
            with open("cache/index.pkl", "wb+") as f:
                pickle.dump(self.index, f)
            with open("cache/docmap.pkl", "wb+") as f:
                pickle.dump(self.docmap, f)
            with open("cache/term_frequencies.pkl", "wb+") as f:
                pickle.dump(self.term_frequencies, f)
            with open("cache/doc_length.pkl", "wb+") as f:
                pickle.dump(self.doc_lengths, f)

        def load(self):
            with open("cache/index.pkl", "rb") as f:
                index = pickle.load(f)
                if index:
                    self.index = index
                else:
                    raise FileNotFoundError
            with open("cache/docmap.pkl", "rb") as f:
                docmap = pickle.load(f)
                if docmap:
                    self.docmap = docmap
                else:
                    raise FileNotFoundError
            with open("cache/term_frequencies.pkl", "rb") as f:
                term_frequencies = pickle.load(f)
                if term_frequencies:
                    self.term_frequencies = term_frequencies
                else:
                    raise FileNotFoundError
            with open("cache/doc_length.pkl", "rb") as f:
                doc_lengths = pickle.load(f)
                if doc_lengths:
                    self.doc_lengths = doc_lengths
                else:
                    raise FileNotFoundError

        def get_tf(self, doc_id, term):
            tokenize = self.__tokenize(term)

            if len(tokenize) > 1:
                raise ValueError("Expected a single token, got multiple")
            if len(tokenize) == 0:
                return 0
            doc = self.term_frequencies.get(doc_id, 0)
            if doc == 0:
                return 0
            else:
                return doc.get(tokenize[0], 0)

        def get_bm25_idf(self, term):
            tokenize = self.__tokenize(term)
            if len(tokenize) > 1:
                raise ValueError("must only be one term")

            N = len(list(self.docmap))
            df = len(list(self.index.get(tokenize[0], 0)))

            bm25 = math.log((N - df + 0.5) / (df + 0.5) + 1)

            return bm25

        def bm25_idf_command(self, term):
            self.load()
            return self.get_bm25_idf(term)

        def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
            doc_length = self.doc_lengths.get(doc_id, 0)
            length_norm = 1 - b + b * (doc_length / self.__get_avg_doc_length())

            tf = self.get_tf(doc_id, term)
            bm25 = (tf * (k1 + 1)) / (tf + k1 * length_norm)
            return bm25

        def bm25_tf_command(self, doc_id, term, k1=BM25_K1, b=BM25_B):
            self.load()
            return self.get_bm25_tf(int(doc_id), term, k1, b)

        def bm25(self, doc_id, term):
            bm25 = self.get_bm25_idf(term) * self.get_bm25_tf(doc_id, term)

            return bm25

        def bm25_search(self, query, limit=5):
            tokenized = self.__tokenize(query)
            scores = {}

            if len(tokenized) == 0:
                return scores

            for key in self.docmap:
                scores[key] = sum(self.bm25(key, token) for token in tokenized)

            scores_sorted = sorted(
                scores.items(), key=lambda item: item[1], reverse=True
            )

            res = {}
            for key, value in scores_sorted[:limit]:
                res[key] = f"{self.docmap[key]['title']} {value:.2f}"

            return res

        def bm25_command(self, query):
            self.load()
            return self.bm25_search(query)

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
