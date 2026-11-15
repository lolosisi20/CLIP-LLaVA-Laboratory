import hashlib
import json
import re
import threading
from dataclasses import dataclass

import httpx
import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings

RRF_K = 60
embedding_cache: dict = {}
cache_lock = threading.Lock()


@dataclass
class Passage:
    text: str
    source: str
    title: str = ""


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def embed(texts: list[str], is_query: bool = False) -> np.ndarray:
    """EmbeddingGemma vectors from the orchestrator, cached on disk, normalised so a dot product is the cosine."""
    prefix = "task: search result | query: " if is_query else "title: none | text: "
    inputs = [prefix + text[:2000] for text in texts]
    cache_path = settings.cache_dir / "embeddings.jsonl"
    cache = load_embedding_cache(cache_path)
    keys = [hashlib.sha1(text.encode()).hexdigest() for text in inputs]
    missing = [(key, text) for key, text in zip(keys, inputs) if key not in cache]
    for start in range(0, len(missing), 32):
        batch = missing[start:start + 32]
        response = httpx.post(settings.embeddings_url, json={"input": [text for _, text in batch]}, timeout=300)
        response.raise_for_status()
        vectors = [item["embedding"] for item in sorted(response.json()["data"], key=lambda item: item["index"])]
        with cache_lock, cache_path.open("a", encoding="utf-8") as file:
            for (key, _), vector in zip(batch, vectors):
                cache[key] = vector
                file.write(json.dumps({"key": key, "vector": vector}) + "\n")
    matrix = np.array([cache[key] for key in keys], dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def load_embedding_cache(path) -> dict:
    if not embedding_cache and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            embedding_cache[item["key"]] = item["vector"]
    return embedding_cache


class HybridIndex:
    """Sparse (BM25) and dense (embedding) search over the same passages, merged by reciprocal rank fusion."""

    def __init__(self, passages: list[Passage], use_embeddings: bool = True):
        self.passages = passages
        self.bm25 = BM25Okapi([words(passage.text) for passage in passages]) if passages else None
        self.vectors = embed([passage.text for passage in passages]) if passages and use_embeddings else None

    def search(self, query: str, k: int = 5) -> list[Passage]:
        if not self.passages:
            return []
        rankings = [np.argsort(-self.bm25.get_scores(words(query)))]
        if self.vectors is not None:
            query_vector = embed([query], is_query=True)[0]
            rankings.append(np.argsort(-(self.vectors @ query_vector)))
        fused = np.zeros(len(self.passages))
        for ranking in rankings:
            fused[ranking] += 1 / (RRF_K + np.arange(1, len(ranking) + 1))
        return [self.passages[index] for index in np.argsort(-fused)[:k]]
