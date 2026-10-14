"""Semantic cache: return a stored answer when a new prompt is *similar enough* to an old one.

Two layers:
1. Exact match  - SHA-256 of the normalized prompt -> O(1) lookup.
2. Semantic     - cosine similarity between embeddings, threshold-gated.

The default embedder is a dependency-free hashing embedder (word + character n-grams hashed
into a fixed-size vector). It catches rephrasings with shared wording ("What's the capital of
France?" vs "what is the capital of france"). For real paraphrase detection plug in a neural
embedder through the `Embedder` protocol (see README roadmap).
"""
from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .schemas import ChatCompletionRequest

_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_CONTRACTIONS = {"what's": "what is", "it's": "it is", "how's": "how is", "who's": "who is",
                 "where's": "where is", "can't": "cannot", "don't": "do not", "i'm": "i am"}
_STOPWORDS = {"the", "a", "an", "please", "can", "you", "me", "tell", "could", "would", "kindly"}


class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray: ...


def normalize(text: str) -> str:
    text = text.lower().strip()
    for k, v in _CONTRACTIONS.items():
        text = text.replace(k, v)
    return " ".join(_WORD.findall(text))


def numbers_in(text: str) -> tuple[str, ...]:
    """Numbers carry meaning that embeddings blur ("17 * 23" vs "17 * 24"), so a semantic hit
    is only allowed when both prompts contain exactly the same numbers."""
    return tuple(_NUMBER.findall(text))


class HashingEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = [w for w in normalize(text).split() if w not in _STOPWORDS]
        features = list(words)
        features += [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]           # word bigrams
        for w in words:                                                     # char trigrams
            padded = f"#{w}#"
            features += [padded[i:i + 3] for i in range(len(padded) - 2)]
        for f in features:
            idx, sign = self._bucket(f)
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


@dataclass
class CacheEntry:
    key: str
    partition: str
    response: dict
    created: float
    numbers: tuple[str, ...] = ()


@dataclass
class CacheResult:
    response: dict
    similarity: float
    kind: str  # "exact" | "semantic"


class VectorIndex:
    """Contiguous matrix of unit vectors -> one matrix-vector product per lookup.

    Deleted rows are zeroed (similarity 0, never matches) and reused, so lookups stay O(n * dim)
    with no per-request allocation of the whole matrix.
    """

    def __init__(self, dim: int, capacity: int = 256):
        self.matrix = np.zeros((capacity, dim), dtype=np.float32)
        self.keys: list[str | None] = []
        self.slot: dict[str, int] = {}
        self.free: list[int] = []

    def add(self, key: str, vec: np.ndarray) -> None:
        if key in self.slot:
            self.matrix[self.slot[key]] = vec
            return
        if self.free:
            i = self.free.pop()
            self.keys[i] = key
        else:
            i = len(self.keys)
            if i == len(self.matrix):
                self.matrix = np.vstack([self.matrix, np.zeros_like(self.matrix)])
            self.keys.append(key)
        self.matrix[i] = vec
        self.slot[key] = i

    def remove(self, key: str) -> None:
        i = self.slot.pop(key, None)
        if i is not None:
            self.matrix[i] = 0
            self.keys[i] = None
            self.free.append(i)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        n = len(self.keys)
        if n == len(self.free):
            return []
        sims = self.matrix[:n] @ query
        k = min(k, n)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.keys[i], float(sims[i])) for i in top if self.keys[i] is not None]


class SemanticCache:
    def __init__(self, embedder: Embedder | None = None, threshold: float = 0.92, ttl_seconds: float = 3600,
                 max_entries: int = 10_000, clock: Callable[[], float] = time.time, dim: int = 256):
        self.embedder = embedder or HashingEmbedder(dim)
        self.dim = dim
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()  # LRU order
        self._indexes: dict[str, VectorIndex] = {}                   # one per model partition
        self.hits = self.semantic_hits = self.misses = 0

    @staticmethod
    def prompt_text(request: ChatCompletionRequest) -> str:
        return "\n".join(f"{m.role}:{m.content}" for m in request.messages)

    @staticmethod
    def _key(partition: str, text: str) -> str:
        return hashlib.sha256(f"{partition}|{normalize(text)}".encode()).hexdigest()

    def _expired(self, e: CacheEntry, now: float) -> bool:
        return now - e.created > self.ttl

    def _delete(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry:
            self._indexes[entry.partition].remove(key)

    def get(self, request: ChatCompletionRequest) -> CacheResult | None:
        now = self._clock()
        text, partition = self.prompt_text(request), request.model
        key = self._key(partition, text)

        exact = self._entries.get(key)
        if exact is not None:
            if not self._expired(exact, now):
                self._entries.move_to_end(key)
                self.hits += 1
                return CacheResult(exact.response, 1.0, "exact")
            self._delete(key)

        index = self._indexes.get(partition)
        if index is not None:
            numbers = numbers_in(text)
            for cand_key, sim in index.search(self.embedder.embed(text)):
                if sim < self.threshold:
                    break
                entry = self._entries[cand_key]
                if self._expired(entry, now):
                    self._delete(cand_key)
                    continue
                if entry.numbers != numbers:
                    continue
                self._entries.move_to_end(cand_key)
                self.hits += 1
                self.semantic_hits += 1
                return CacheResult(entry.response, sim, "semantic")
        self.misses += 1
        return None

    def put(self, request: ChatCompletionRequest, response: dict) -> None:
        text, partition = self.prompt_text(request), request.model
        key = self._key(partition, text)
        self._entries[key] = CacheEntry(key, partition, response, self._clock(), numbers_in(text))
        self._entries.move_to_end(key)
        self._indexes.setdefault(partition, VectorIndex(self.dim)).add(key, self.embedder.embed(text))
        while len(self._entries) > self.max_entries:
            self._delete(next(iter(self._entries)))

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"entries": len(self), "hits": self.hits, "semantic_hits": self.semantic_hits,
                "misses": self.misses, "hit_rate": round(self.hits / total, 4) if total else 0.0}
