"""Build the search index.

    python -m backend.ingest                # 50,000 articles (about 5 minutes)
    python -m backend.ingest --n 120000     # full AG News training set
    python -m backend.ingest --index flat   # exact search instead of HNSW
    python -m backend.ingest --csv data/raw/ag_news_train.csv   # use a local file

If data/raw/ag_news_train.csv exists it is used automatically (get it with scripts/download_dataset.sh).
"""
import argparse
import csv
import html
import json
import pickle
import random
import re
import time
from pathlib import Path

import faiss
import numpy as np

from . import config
from .bm25 import BM25Index


def clean(text: str) -> str:
    text = html.unescape(text).replace("\\", " ")
    return re.sub(r"\s+", " ", text).strip()


def load_csv(path, n=None, seed=42):
    """Read the AG News CSV format: label (1-4), title, description. Returns (texts, labels 0-3)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            label = int(row[0]) - 1
            rows.append((label, clean(f"{row[1]}. {' '.join(row[2:])}")))
    random.Random(seed).shuffle(rows)
    if n:
        rows = rows[:n]
    return [text for _, text in rows], [label for label, _ in rows]


def make_faiss_index(emb: np.ndarray, index_type: str):
    dim = emb.shape[1]
    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index


def build(texts, labels, model, index_type="hnsw", out_dir=config.DATA_DIR, batch_size=128):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[1/3] Embedding {len(texts):,} articles ...")
    emb = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype("float32")
    np.save(out / "embeddings.npy", emb)

    print(f"[2/3] Building {index_type.upper()} vector index ...")
    faiss.write_index(make_faiss_index(emb, index_type), str(out / "index.faiss"))

    print("[3/3] Building BM25 keyword index ...")
    BM25Index().fit(texts).save(out)

    with open(out / "docs.pkl", "wb") as f:
        pickle.dump({"texts": list(texts), "labels": np.asarray(labels, dtype=np.int8)}, f)

    meta = {
        "documents": len(texts), "dimensions": int(emb.shape[1]), "model": config.MODEL_NAME,
        "index_type": index_type, "build_seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done in {meta['build_seconds']} s. Start the app with ./run.sh")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000, help="number of articles to index (max 120000)")
    ap.add_argument("--index", choices=["hnsw", "flat"], default="hnsw")
    ap.add_argument("--csv", help="path to an AG News style CSV (label, title, description)")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    csv_path = Path(args.csv) if args.csv else config.DATA_DIR / "raw" / "ag_news_train.csv"
    if csv_path.exists():
        print(f"Loading dataset from {csv_path} ...")
        texts, labels = load_csv(csv_path, args.n)
    else:
        from datasets import load_dataset
        print("Local file not found, downloading AG News from Hugging Face ...")
        try:
            ds = load_dataset("fancyzhx/ag_news", split="train")
        except Exception:
            ds = load_dataset("ag_news", split="train")
        ds = ds.shuffle(seed=42).select(range(min(args.n, len(ds))))
        texts = [clean(t) for t in ds["text"]]
        labels = ds["label"]
    print(f"{len(texts):,} articles loaded")

    model = SentenceTransformer(config.MODEL_NAME)
    build(texts, labels, model, args.index)


if __name__ == "__main__":
    main()
