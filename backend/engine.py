"""Hybrid search engine: BM25 keywords + dense embeddings (FAISS) + Reciprocal Rank Fusion."""
import json
import pickle
import threading
import time
from collections import OrderedDict
from pathlib import Path

import faiss
import numpy as np

from . import config
from .bm25 import BM25Index


def reciprocal_rank_fusion(rankings, k: int = config.RRF_K):
    """Merge several ranked id lists. A document ranked high in ANY list scores well."""
    scores = {}
    for ids in rankings:
        for pos, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + pos + 1)
    order = sorted(scores, key=scores.__getitem__, reverse=True)
    return order, scores


class SearchEngine:
    DEPTH = 100  # candidates pulled from each retriever before fusion

    def __init__(self, data_dir=config.DATA_DIR, model=None):
        data_dir = Path(data_dir)
        if not (data_dir / "meta.json").exists():
            raise RuntimeError("Search index not found. Build it first with:  python -m backend.ingest")
        self.meta = json.loads((data_dir / "meta.json").read_text())
        with open(data_dir / "docs.pkl", "rb") as f:
            docs = pickle.load(f)
        self.texts = docs["texts"]
        self.labels = np.asarray(docs["labels"])
        self.embeddings = np.load(data_dir / "embeddings.npy", mmap_mode="r")
        self.index = faiss.read_index(str(data_dir / "index.faiss"))
        self.bm25 = BM25Index.load(data_dir)
        self.model = model or self._load_model(self.meta["model"])
        self._lock = threading.Lock()
        self._cache = OrderedDict()
        self._encode("warm up")

    @staticmethod
    def _load_model(name):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(name)

    # ---------- helpers ----------
    def _encode(self, query: str) -> np.ndarray:
        with self._lock:
            if query in self._cache:
                self._cache.move_to_end(query)
                return self._cache[query]
            vec = self.model.encode([query], normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False).astype("float32")
            self._cache[query] = vec
            if len(self._cache) > 512:
                self._cache.popitem(last=False)
            return vec

    def _mask(self, category):
        if not category or category.lower() in ("all", ""):
            return None
        names = [c.lower() for c in config.CATEGORIES]
        if category.lower() not in names:
            raise ValueError(f"Unknown topic '{category}'. Choose from: {', '.join(config.CATEGORIES)}")
        return self.labels == names.index(category.lower())

    def _keyword(self, query, mask):
        return self.bm25.top(query, self.DEPTH, mask)

    def _semantic(self, query, mask):
        fetch = self.DEPTH if mask is None else min(len(self.texts), self.DEPTH * 8)
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = max(64, fetch)
        scores, ids = self.index.search(self._encode(query), fetch)
        ids, scores = ids[0], scores[0]
        keep = ids >= 0
        ids, scores = ids[keep], scores[keep]
        if mask is not None:
            keep = mask[ids]
            ids, scores = ids[keep], scores[keep]
        return ids[: self.DEPTH].tolist(), scores[: self.DEPTH].tolist()

    def _item(self, doc_id, score, kw_pos, sem_pos):
        return {
            "id": int(doc_id),
            "text": self.texts[doc_id],
            "category": config.CATEGORIES[int(self.labels[doc_id])],
            "score": round(float(score), 4),
            "ranks": {"keyword": kw_pos.get(doc_id), "semantic": sem_pos.get(doc_id)},
        }

    def _assemble(self, mode, k, kw, sem):
        kw_ids, kw_scores = kw
        sem_ids, sem_scores = sem
        kw_pos = {d: i + 1 for i, d in enumerate(kw_ids)}
        sem_pos = {d: i + 1 for i, d in enumerate(sem_ids)}
        if mode == "keyword":
            pairs = list(zip(kw_ids, kw_scores))[:k]
        elif mode == "semantic":
            pairs = list(zip(sem_ids, sem_scores))[:k]
        else:
            order, fused = reciprocal_rank_fusion([kw_ids, sem_ids])
            pairs = [(d, fused[d]) for d in order[:k]]
        return [self._item(d, s, kw_pos, sem_pos) for d, s in pairs]

    # ---------- public API ----------
    def search(self, query, mode="hybrid", k=10, category=None):
        t0 = time.perf_counter()
        mask = self._mask(category)
        terms, kw, sem = [], ([], []), ([], [])
        if mode in ("keyword", "hybrid"):
            *kw, terms = self._keyword(query, mask)
        if mode in ("semantic", "hybrid"):
            sem = self._semantic(query, mask)
        results = self._assemble(mode, k, tuple(kw), sem)
        return {"query": query, "mode": mode, "terms": terms, "results": results,
                "took_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def compare(self, query, k=10, category=None):
        t0 = time.perf_counter()
        mask = self._mask(category)
        *kw, terms = self._keyword(query, mask)
        kw = tuple(kw)
        sem = self._semantic(query, mask)
        out = {m: {"results": self._assemble(m, k, kw, sem)} for m in ("keyword", "semantic", "hybrid")}
        shared = {r["id"] for r in out["keyword"]["results"]} & {r["id"] for r in out["semantic"]["results"]}
        out.update(query=query, terms=terms, overlap=len(shared),
                   took_ms=round((time.perf_counter() - t0) * 1000, 1))
        return out

    def similar(self, doc_id, k=10):
        if not 0 <= doc_id < len(self.texts):
            raise ValueError("Unknown article id")
        t0 = time.perf_counter()
        vec = np.asarray(self.embeddings[doc_id], dtype="float32")[None, :]
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = 64
        scores, ids = self.index.search(vec, k + 1)
        pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0 and i != doc_id][:k]
        results = [self._item(i, s, {}, {}) for i, s in pairs]
        return {"source": {"id": doc_id, "text": self.texts[doc_id]}, "results": results,
                "took_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def stats(self):
        return {"articles": len(self.texts), "model": self.meta["model"].split("/")[-1],
                "index": self.meta["index_type"], "dimensions": self.meta["dimensions"],
                "topics": config.CATEGORIES}
