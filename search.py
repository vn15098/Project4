import pickle
from pathlib import Path

import faiss
import numpy as np

from featureExtraction import extract_features

artifacts_dir = Path(__file__).resolve().parent / "artifacts"
filenames_path = artifacts_dir / "filenames.pkl"
index_path = artifacts_dir / "faiss_index.bin"

missing_artifacts = [
    path for path in (filenames_path, index_path) if not path.is_file()
]
if missing_artifacts:
    missing_files = ", ".join(str(path) for path in missing_artifacts)
    raise FileNotFoundError(
        f"Required search artifacts are missing: {missing_files}. "
        "Run embedding.py and then faizz.py from the project."
    )

with filenames_path.open("rb") as source:
    filenames = pickle.load(source)

index = faiss.read_index(str(index_path))
if index.ntotal != len(filenames):
    raise ValueError(
        f"Index contains {index.ntotal} vectors but {len(filenames)} filenames "
        f"were loaded from {filenames_path}. Rebuild the embeddings and index."
    )


def get_similar_products(image_path, top_k=5):
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if index.ntotal == 0:
        return []

    query_embedding = extract_features(
        image_path
    )

    query_embedding = np.asarray([query_embedding], dtype=np.float32)
    query_path = Path(image_path).resolve()
    matching_indices = [
        i for i, filename in enumerate(filenames)
        if Path(filename).resolve() == query_path
    ]
    excluded_index = matching_indices[0] if matching_indices else None
    search_count = min(top_k + int(excluded_index is not None), index.ntotal)

    scores, indices = index.search(
        query_embedding,
        search_count
    )

    recommendations = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx == excluded_index:
            continue
        recommendations.append({"path": filenames[idx], "score": float(score)})
        if len(recommendations) == top_k:
            break

    return recommendations


def evaluate_retrieval(retrieved_ids, relevant_ids, k):
    """Return Precision@K and Recall@K for a retrieval result list."""
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids must contain at least one relevant item.")

    top_k = retrieved_ids[:k]
    relevant_retrieved = len(set(top_k) & relevant)
    return {
        "precision_at_k": relevant_retrieved / k,
        "recall_at_k": relevant_retrieved / len(relevant),
    }