#!/usr/bin/env python3

import argparse
import os
import json
import nltk
import ssl
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter


    
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser = subparsers.add_parser("build")

    args = parser.parse_args()

    # Fix SSL on macOS
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('stopwords', quiet=True)

    class InvertedIndex:
        def __init__(self):
            self.index = {}
            self.docmap = {}
            self.term_frequencies = Counter()
        
        def __add_document(self, doc_id, text):
            searchTerm = ''
            for n in text:
                if n.isalnum() or n == " ":
                    searchTerm += n

            tokenize = searchTerm.split()

            for token in tokenize:
                token = token.lower()
                if token not in self.index:
                    self.index[token] = set()
                else: 
                    self.index[token].add(doc_id)
                if token not in self.term_frequencies: # stoping here
                    self.term_frequencies[doc_id] = set()
                else:
                    self.term_frequencies[doc_id] += 1
        
        def get_documents(self, term):
            term = term.lower()
            ids = sorted(self.index.get(term, set()))
            return ids

        def build(self):
            with open("./data/movies.json", "r") as file:
                data = json.load(file)

            for key, value in data.items():
                for i, m in enumerate(value, start=1):
                    self.__add_document(i,f"{m['title']} {m['description']}")
                    if i not in self.docmap:
                        self.docmap[i] = m
        def save(self):
            if not os.path.isdir('cache'):
                os.mkdir('cache')
            with open('cache/index.pkl', 'wb+') as f:
                pickle.dump(self.index, f)
            with open('cache/docmap.pkl', 'wb+') as f:
                pickle.dump(self.docmap, f)
            with open('cache/term_frequencies.pkl', 'wb+') as f:
                pickle.dump(self.term_frequencies, f)

        def load(self):
                with open('cache/index.pkl', 'rb') as f:
                    index = pickle.load(f)
                    if index:
                        self.index = index
                    else: 
                        raise FileNotFoundError 
                with open('cache/docmap.pkl', 'rb') as f:
                    docmap = pickle.load(f)
                    if docmap:
                        self.docmap = docmap
                    else: 
                        raise FileNotFoundError 
                with open('cache/term_frequencies.pkl', 'rb') as f:
                    term_frequencies = pickle.load(f)
                    if docmap:
                        self.term_frequencies = term_frequencies
                    else: 
                        raise FileNotFoundError 
                

    match args.command:
        case "search":
            i = InvertedIndex()
            try:
                i.load()
                print(i.docmap)
                token = args.command.split()
                docs = dict()
                for token in list(i.index)[:5]:
                    docs_ids = list(i.index[token])
                    for docs_id in docs_ids:
                        docs[docs_id] = i.docmap[docs_id]
            except FileNotFoundError:
                print("File not found") 
            string = ''
            for doc in docs.values():
                string += f"{doc['id']} {doc['title']}\n"
            # print(string)   
            pass
        case "build":
            i = InvertedIndex()
            i.build()
            i.save()
            
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()