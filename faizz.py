import pickle
from pathlib import Path

import numpy as np
import faiss

artifacts_dir = Path(__file__).resolve().parent / "artifacts"
embeddings_path = artifacts_dir / "embeddings.pkl"

with embeddings_path.open("rb") as source:
    embeddings = pickle.load(source)

if not embeddings:
    raise ValueError(f"No embeddings found in {embeddings_path}. Run embedding.py first.")

embedding_array = np.asarray(embeddings, dtype="float32")
if embedding_array.ndim != 2 or not np.isfinite(embedding_array).all():
    raise ValueError("Embeddings must be a finite two-dimensional array.")

norms = np.linalg.norm(embedding_array, axis=1)
if np.any(norms == 0):
    raise ValueError("Embeddings must be non-zero vectors for cosine similarity.")
embedding_array /= norms[:, np.newaxis]

dimension = embedding_array.shape[1]

index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 80
index.hnsw.efSearch = 64

index.add(embedding_array)

faiss.write_index(
    index,
    str(artifacts_dir / "faiss_index.bin")
)

print("FAISS Index Saved")