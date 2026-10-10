"""Qdrant-backed memory store.

One Qdrant *point* = one memory:
    id       -> memory id (UUID)
    vector   -> embedding of the memory text
    payload  -> everything else (text, user_id, timestamps, validity, provenance)

So there is a single source of truth; no separate SQL table to keep in sync.
"""
import logging
from collections.abc import Iterator

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .models import Memory, ScoredMemory, utcnow

log = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int):
        self.client = client
        self.collection = collection
        self.dim = dim

    # ------------------------------------------------------------------ setup
    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            existing = info.config.params.vectors.size  # type: ignore[union-attr]
            if existing != self.dim:
                raise RuntimeError(
                    f"Collection '{self.collection}' has vector size {existing}, "
                    f"but EMBEDDING_DIM={self.dim}. Use a new collection name or matching model.")
            return
        self.client.create_collection(
            self.collection,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
        )
        # Payload indexes make the per-user / active-only filters fast at scale.
        for field, schema in (("user_id", qm.PayloadSchemaType.KEYWORD),
                              ("is_active", qm.PayloadSchemaType.BOOL)):
            self.client.create_payload_index(self.collection, field, field_schema=schema)
        log.info("collection_created", extra={"ctx": {"collection": self.collection, "dim": self.dim}})

    def ping(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---------------------------------------------------------------- filters
    @staticmethod
    def _user_filter(user_id: str, active_only: bool = True) -> qm.Filter:
        must: list[qm.Condition] = [qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))]
        if active_only:
            must.append(qm.FieldCondition(key="is_active", match=qm.MatchValue(value=True)))
        return qm.Filter(must=must)

    # ----------------------------------------------------------------- writes
    def insert(self, memory: Memory, vector: list[float]) -> Memory:
        self.client.upsert(self.collection, points=[
            qm.PointStruct(id=memory.id, vector=vector, payload=memory.to_payload())])
        return memory

    def set_fields(self, memory_id: str, **fields) -> None:
        payload = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in fields.items()}
        payload["updated_at"] = utcnow().isoformat()
        self.client.set_payload(self.collection, payload=payload, points=[memory_id])

    def invalidate(self, memory_id: str, superseded_by: str | None = None) -> None:
        """Soft delete: keeps the memory as history but excludes it from recall."""
        self.set_fields(memory_id, is_active=False, valid_until=utcnow(), superseded_by=superseded_by)

    def confirm(self, memory: Memory) -> None:
        """User re-stated a known fact: strengthen it instead of storing a duplicate."""
        self.set_fields(memory.id, confirmation_count=memory.confirmation_count + 1)

    def touch(self, memories: list[Memory]) -> None:
        """Record usage. Read-modify-write; small races only affect statistics."""
        now = utcnow()
        for m in memories:
            self.set_fields(m.id, access_count=m.access_count + 1, last_accessed_at=now)
 
    def delete(self, memory_ids: list[str]) -> None:
        if memory_ids:
            self.client.delete(self.collection, points_selector=qm.PointIdsList(points=memory_ids))

    # ------------------------------------------------------------------ reads
    def search(self, user_id: str, vector: list[float], limit: int,
               min_score: float | None = None) -> list[ScoredMemory]:
        result = self.client.query_points(
            self.collection, query=vector, query_filter=self._user_filter(user_id),
            limit=limit, score_threshold=min_score, with_payload=True,
        )
        return [ScoredMemory(memory=Memory.from_payload(p.id, p.payload or {}), score=p.score)
                for p in result.points]

    def get(self, memory_id: str) -> Memory | None:
        points = self.client.retrieve(self.collection, ids=[memory_id], with_payload=True)
        return Memory.from_payload(points[0].id, points[0].payload or {}) if points else None

    def iter_memories(self, user_id: str, active_only: bool = True,
                      with_vectors: bool = False) -> Iterator[tuple[Memory, list[float] | None]]:
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, scroll_filter=self._user_filter(user_id, active_only),
                limit=256, offset=offset, with_payload=True, with_vectors=with_vectors,
            )
            for p in points:
                yield Memory.from_payload(p.id, p.payload or {}), (p.vector if with_vectors else None)  # type: ignore[misc]
            if offset is None:
                break

    def list(self, user_id: str, active_only: bool = True) -> list[Memory]:
        return sorted((m for m, _ in self.iter_memories(user_id, active_only)),
                      key=lambda m: m.created_at)

    def count(self, user_id: str, active_only: bool = True) -> int:
        return self.client.count(self.collection, count_filter=self._user_filter(user_id, active_only),
                                 exact=True).count
