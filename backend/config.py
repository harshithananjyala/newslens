from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FRONTEND_DIR = ROOT / "frontend"
BENCH_DIR = ROOT / "benchmarks"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]  # AG News label order
RRF_K = 60  # Reciprocal Rank Fusion constant
