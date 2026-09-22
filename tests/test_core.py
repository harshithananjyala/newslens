import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.bm25 import BM25Index
from backend.engine import SearchEngine, reciprocal_rank_fusion
from backend.ingest import build
from backend.main import create_app

TEXTS = [
    "Oil prices climb as crude supply tightens",
    "Stocks fall on weak quarterly earnings from retailers",
    "Striker scores late goal to win the championship final",
    "Scientists discover a distant planet orbiting a nearby star",
    "New software vulnerability lets attackers steal data",
    "Central bank raises interest rates to fight inflation",
    "Tennis champion beats rival in straight sets",
    "Government leaders meet to negotiate peace agreement",
]
LABELS = [2, 2, 1, 3, 3, 2, 1, 0]


class FakeModel:
    """Deterministic hashing embedder so tests don't need to download a real model."""
    def encode(self, texts, **kwargs):
        out = np.zeros((len(texts), 64), dtype="float32")
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, hash(w) % 64] += 1.0
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out


def test_bm25_ranks_matching_doc_first():
    idx = BM25Index().fit(TEXTS)
    ids, scores, terms = idx.top("oil prices", 3)
    assert ids[0] == 0 and "oil" in terms
    assert scores == sorted(scores, reverse=True)


def test_bm25_unknown_terms_return_nothing():
    ids, scores, terms = BM25Index().fit(TEXTS).top("zzzzqqq", 3)
    assert ids == [] and terms == []


def test_bm25_save_load_roundtrip(tmp_path):
    idx = BM25Index().fit(TEXTS)
    idx.save(tmp_path)
    assert BM25Index.load(tmp_path).top("peace agreement", 1)[0] == idx.top("peace agreement", 1)[0]


def test_rrf_rewards_agreement():
    order, scores = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    assert order[0] == 1 and set(order) == {1, 2, 3, 4}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    build(TEXTS, LABELS, FakeModel(), "hnsw", d)
    engine = SearchEngine(d, model=FakeModel())
    with TestClient(create_app(engine)) as c:
        yield c


def test_search_modes(client):
    for mode in ("keyword", "semantic", "hybrid"):
        r = client.get("/api/search", params={"q": "oil prices", "mode": mode, "k": 3}).json()
        assert r["results"] and r["took_ms"] >= 0
    assert client.get("/api/search", params={"q": "oil prices"}).json()["results"][0]["id"] == 0


def test_topic_filter(client):
    r = client.get("/api/search", params={"q": "champion", "category": "Sports"}).json()
    assert r["results"] and all(x["category"] == "Sports" for x in r["results"])
    assert client.get("/api/search", params={"q": "x", "category": "Nope"}).status_code == 400


def test_compare_and_similar(client):
    c = client.get("/api/compare", params={"q": "interest rates", "k": 3}).json()
    assert set(c) >= {"keyword", "semantic", "hybrid", "overlap", "terms"}
    s = client.get("/api/similar/0", params={"k": 3}).json()
    assert all(x["id"] != 0 for x in s["results"])
    assert client.get("/api/similar/999").status_code == 404


def test_stats_and_frontend(client):
    assert client.get("/api/stats").json()["articles"] == len(TEXTS)
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
