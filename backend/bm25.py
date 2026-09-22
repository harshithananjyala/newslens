"""A fast BM25 keyword index built on scipy sparse matrices.

All BM25 term weights are pre-computed once at index time, so scoring a query is
just "sum the columns for the query terms" - a few milliseconds even for 100K+ docs.
"""
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = None
        self.weights = None  # CSC matrix: documents x terms
        self._vocab = None

    def fit(self, texts):
        self.vectorizer = CountVectorizer(
            lowercase=True,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z0-9]{2,}\b",
            dtype=np.float32,
        )
        tf = self.vectorizer.fit_transform(texts).tocsr()
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = float(doc_len.mean())
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        norm = self.k1 * (1.0 - self.b + self.b * doc_len[rows] / avgdl)
        tf.data = (tf.data * (self.k1 + 1.0) / (tf.data + norm) * idf[tf.indices]).astype(np.float32)
        self.weights = tf.tocsc()
        self._vocab = self.vectorizer.get_feature_names_out()
        return self

    def score(self, query: str):
        """Return (scores for every document, matched query terms)."""
        n_docs = self.weights.shape[0]
        cols = self.vectorizer.transform([query]).indices
        if len(cols) == 0:
            return np.zeros(n_docs, dtype=np.float32), []
        scores = np.asarray(self.weights[:, cols].sum(axis=1)).ravel()
        return scores, [str(self._vocab[c]) for c in cols]

    def top(self, query: str, n: int, mask=None):
        """Top-n document ids, their scores, and the matched terms."""
        scores, terms = self.score(query)
        if mask is not None:
            scores = np.where(mask, scores, 0.0)
        n = max(1, min(n, scores.size))
        idx = np.argpartition(-scores, n - 1)[:n]
        idx = idx[np.argsort(-scores[idx])]
        idx = idx[scores[idx] > 0]
        return idx.tolist(), scores[idx].tolist(), terms

    def save(self, directory):
        directory = Path(directory)
        sparse.save_npz(directory / "bm25.npz", self.weights)
        with open(directory / "vectorizer.pkl", "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "k1": self.k1, "b": self.b}, f)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with open(directory / "vectorizer.pkl", "rb") as f:
            blob = pickle.load(f)
        obj = cls(blob["k1"], blob["b"])
        obj.vectorizer = blob["vectorizer"]
        obj.weights = sparse.load_npz(directory / "bm25.npz").tocsc()
        obj._vocab = obj.vectorizer.get_feature_names_out()
        return obj
