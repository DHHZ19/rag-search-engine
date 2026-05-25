import json
import math
import os
import pickle
from collections import Counter

from constants import BM25_B, BM25_K1
from nltk.stem import PorterStemmer
from search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    SearchResult,
    format_search_result,
)


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
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

    def bm25_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SearchResult]:
        query_tokens = self.__tokenize(query)
        limit = int(limit)

        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score

        if len(scores.items()) >= limit:
            limit = math.floor(limit / 100)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)

        return results

    def bm25_command(self, query):
        self.load()
        return self.bm25_search(query)
