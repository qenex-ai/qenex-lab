#!/usr/bin/env python3
"""
QENEX LAB Semantic Cache
Intelligent caching with embedding-based similarity search
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import redis
import hashlib
import json
from typing import Optional


class SemanticCache:
    """
    Two-tier caching system:
    1. Redis - exact hash matches (fastest)
    2. FAISS - semantic similarity matches (embeddings)
    """

    def __init__(self):
        # Exact match cache (Redis)
        try:
            self.redis = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            # Test connection
            self.redis.ping()
            print("[Semantic Cache] ✓ Redis connected")
        except Exception as e:
            print(f"[Semantic Cache] ⚠️  Redis connection failed: {e}")
            self.redis = None

        # Semantic similarity cache
        print("[Semantic Cache] Loading sentence-transformers model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # 80MB model
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)  # L2 distance for similarity
        self.cached_queries = []  # Store original queries
        self.cached_responses = []  # Store responses

        # Similarity threshold (0.0 = identical, higher = more different)
        self.similarity_threshold = 0.3  # Tune based on testing

        print(f"[Semantic Cache] ✓ Initialized (dimension={self.dimension}, threshold={self.similarity_threshold})")

    def _hash_query(self, query: str) -> str:
        """Generate hash for exact matching"""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def get(self, query: str) -> Optional[dict]:
        """Try exact match first, then semantic similarity"""
        # 1. Exact match (Redis) - fastest
        if self.redis:
            query_hash = self._hash_query(query)
            try:
                cached = self.redis.get(f"exact:{query_hash}")
                if cached:
                    print(f"[Cache] ✅ EXACT HIT: {query[:50]}")
                    return json.loads(cached)
            except Exception as e:
                print(f"[Cache] Redis error: {e}")

        # 2. Semantic similarity (FAISS)
        if self.index.ntotal == 0:
            print(f"[Cache] ❌ MISS: {query[:50]} (no cached embeddings)")
            return None  # No cached embeddings yet

        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, k=1)

        if distances[0][0] < self.similarity_threshold:
            idx = indices[0][0]
            print(f"[Cache] ✅ SEMANTIC HIT: distance={distances[0][0]:.3f}")
            print(f"  Original: {self.cached_queries[idx][:50]}")
            print(f"  Query: {query[:50]}")
            return self.cached_responses[idx]

        print(f"[Cache] ❌ MISS: {query[:50]} (distance={distances[0][0]:.3f} > {self.similarity_threshold})")
        return None

    def set(self, query: str, response: dict, model: str):
        """Cache response with both exact and semantic indexing"""
        # 1. Exact cache (Redis) - 1 hour TTL
        if self.redis:
            query_hash = self._hash_query(query)
            try:
                self.redis.setex(
                    f"exact:{query_hash}",
                    3600,  # 1 hour
                    json.dumps(response)
                )
            except Exception as e:
                print(f"[Cache] Redis set error: {e}")

        # 2. Semantic cache (FAISS + in-memory)
        query_embedding = self.model.encode([query])
        self.index.add(query_embedding)
        self.cached_queries.append(query)
        self.cached_responses.append(response)

        print(f"[Cache] 💾 STORED: {query[:50]} -> {model} (total cached: {self.index.ntotal})")

    def clear(self):
        """Clear all caches"""
        if self.redis:
            self.redis.flushdb()
        self.index.reset()
        self.cached_queries = []
        self.cached_responses = []
        print("[Cache] 🗑️  All caches cleared")

    def stats(self) -> dict:
        """Get cache statistics"""
        return {
            "redis_connected": self.redis is not None,
            "faiss_total": self.index.ntotal,
            "model": "all-MiniLM-L6-v2",
            "dimension": self.dimension,
            "threshold": self.similarity_threshold
        }
