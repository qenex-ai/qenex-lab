#!/usr/bin/env python3
"""
QENEX LAB Global Context Bridge
Indexes and searches all knowledge artifacts (discoveries, research files, code)
OMNI_INTEGRATION v1.4.0-INFINITY
"""

from sentence_transformers import SentenceTransformer
import faiss
import json
import os
import numpy as np
from pathlib import Path
from glob import glob
from typing import Dict, List
import aiohttp


class GlobalContextBridge:
    """Indexes and searches all QENEX LAB knowledge artifacts"""

    DISCOVERY_PATHS = [
        "/opt/qenex/scout-cli/discoveries/",
        "/opt/qenex/scout-cli/SYSTEM_MANIFEST.json",
        "/opt/qenex/qlang/spec/QLANG_SPECIFICATION.md",
        "/opt/qenex/brain/README.md",
        "/opt/qenex_lab/README.md",
        "/home/ubuntu/.opencode/agent/qenex-lab.md",
        "/tmp/*.md",  # All markdown reports
    ]

    def __init__(self):
        print("[Context Bridge] Initializing Global Context Bridge...")

        # Embedding model (reuse from semantic_cache)
        print("[Context Bridge] Loading sentence-transformers model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Vector index
        self.dimension = 384
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine sim

        # Document store
        self.documents: List[Dict] = []  # List of {path, content, metadata}
        self.embeddings: List[np.ndarray] = []

        # Julia RAG server URL (optional integration)
        self.rag_server_url = "http://localhost:8891"
        self.rag_available = False

        # Index all files
        self._index_all_files()

        print(f"[Context Bridge] ✓ Indexed {len(self.documents)} documents")
        print(f"[Context Bridge] ✓ FAISS index size: {self.index.ntotal} vectors")

    def _index_all_files(self):
        """Index all discovery files and create embeddings"""
        for pattern in self.DISCOVERY_PATHS:
            if "*" in pattern:
                # Glob pattern
                files = glob(pattern, recursive=True)
            else:
                # Single file or directory
                files = [pattern]

            for filepath in files:
                if os.path.isdir(filepath):
                    # Recursively index directory
                    for root, dirs, filenames in os.walk(filepath):
                        for filename in filenames:
                            if filename.endswith(('.json', '.md', '.txt', '.rst')):
                                self._index_file(os.path.join(root, filename))
                elif os.path.isfile(filepath):
                    self._index_file(filepath)

    def _index_file(self, filepath: str):
        """Index a single file"""
        try:
            # Read file
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Skip empty files
            if not content.strip():
                return

            # Create embedding
            embedding = self.model.encode([content])[0]

            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)

            # Store document
            doc_metadata = {
                'path': filepath,
                'content': content,
                'name': os.path.basename(filepath),
                'size': len(content),
                'type': filepath.split('.')[-1] if '.' in filepath else 'unknown'
            }

            self.documents.append(doc_metadata)
            self.embeddings.append(embedding)

            # Add to FAISS
            self.index.add(np.array([embedding]))

            print(f"[Context Bridge] Indexed: {filepath} ({len(content)} bytes)")

        except Exception as e:
            print(f"[Context Bridge] Failed to index {filepath}: {e}")

    async def gather_context(self, query: str, top_k: int = 3) -> dict:
        """Gather relevant context for a query"""

        print(f"[Context Bridge] Gathering context for query: {query[:50]}...")

        # Create query embedding
        query_embedding = self.model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search FAISS
        if self.index.ntotal > 0:
            # Search for top_k results
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(np.array([query_embedding]), k=k)

            # Retrieve documents
            relevant_docs = []
            for idx, score in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['relevance_score'] = float(score)
                    # Don't include full content in result (too large)
                    doc['content_preview'] = doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content']
                    relevant_docs.append(doc)

            print(f"[Context Bridge] Found {len(relevant_docs)} relevant documents")
        else:
            relevant_docs = []
            print("[Context Bridge] No documents indexed yet")

        # Optional: Query Julia RAG server for additional context
        rag_results = await self._query_rag_server(query) if self.rag_available else []

        return {
            'query': query,
            'discovery_files': relevant_docs,
            'rag_results': rag_results,
            'total_docs': len(self.documents)
        }

    async def _query_rag_server(self, query: str) -> List[dict]:
        """Query Julia RAG server for additional context"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.rag_server_url}/query",
                    json={'query': query},
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('results', [])
        except Exception as e:
            print(f"[Context Bridge] RAG server query failed: {e}")
        return []

    def build_system_prompt(self, context: dict, base_prompt: str = "") -> str:
        """Build system prompt with discovery context"""

        prompt = base_prompt if base_prompt else """You are QENEX LAB v1.4.0-INFINITY, operating in OMNI-AWARE mode.

"""

        # Add discovery context
        if context['discovery_files']:
            prompt += "\nYou have access to the following historical knowledge and discoveries:\n\n"

            for i, doc in enumerate(context['discovery_files'], 1):
                prompt += f"### Discovery {i}: {doc['name']} (relevance: {doc['relevance_score']:.3f})\n"
                prompt += f"Path: {doc['path']}\n"

                # Use full content for system prompt (will be truncated later if needed)
                content = doc['content']
                # Truncate very long documents (reduced for performance)
                if len(content) > 1000:
                    content = content[:1000] + "\n\n[... truncated for length ...]"

                prompt += f"{content}\n\n"

        # Add instructions
        prompt += """
OMNI-AWARE INSTRUCTIONS:
1. Reference these discoveries when relevant to the user's query
2. Cite specific files (by name) when making claims
3. Maintain scientific rigor and physics constraints from the Unified Lagrangian
4. Think step-by-step before responding
5. Draw connections between historical discoveries and current query

"""

        return prompt

    def get_stats(self) -> dict:
        """Get context bridge statistics"""
        return {
            'total_documents': len(self.documents),
            'faiss_vectors': self.index.ntotal,
            'embedding_dimension': self.dimension,
            'rag_server': 'available' if self.rag_available else 'unavailable',
            'sample_files': [doc['name'] for doc in self.documents[:5]]
        }
