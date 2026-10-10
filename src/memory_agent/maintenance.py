"""Background maintenance — keeps memory small and clean over months of use.

consolidate(): find clusters of near-duplicate active memories and merge each into a few facts.
    1. For every active memory, ANN-search its neighbours above CONSOLIDATION_THRESHOLD  (edges)
    2. Union-Find over those edges -> connected components                               (clusters)
    3. Centroid check: drop members far from the cluster's centre (guards against chaining A~B~C)
    4. LLM merges each cluster -> new memories with source_ids; originals are invalidated
purge(): hard-delete memories that were invalidated more than PURGE_AFTER_DAYS ago.
"""
import logging
import math
from datetime import timedelta

from . import prompts
from .config import Settings
from .llm import LLMClient, LLMResponseError, parse_json
from .memory_manager import _coerce_fact
from .models import ClusterPlan, Fact, MaintenanceReport, Memory, utcnow
from .store import MemoryStore

log = logging.getLogger(__name__)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> list[list[str]]:
        out: dict[str, list[str]] = {}
        for x in self.parent:
            out.setdefault(self.find(x), []).append(x)
        return list(out.values())


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class Maintenance:
    def __init__(self, store: MemoryStore, llm: LLMClient, settings: Settings):
        self.store, self.llm, self.s = store, llm, settings

    def find_clusters(self, user_id: str) -> list[list[tuple[Memory, list[float]]]]:
        items = {m.id: (m, v) for m, v in self.store.iter_memories(user_id, with_vectors=True) if v}
        uf = UnionFind()
        for mem_id, (_, vec) in items.items():
            for hit in self.store.search(user_id, vec, limit=self.s.max_cluster_size,
                                         min_score=self.s.consolidation_threshold):
                if hit.memory.id != mem_id and hit.memory.id in items:
                    uf.union(mem_id, hit.memory.id)

        clusters = []
        for ids in uf.groups():
            if len(ids) < 2:
                continue
            members = [items[i] for i in ids]
            dim = len(members[0][1])
            centroid = [sum(v[d] for _, v in members) / len(members) for d in range(dim)]
            core = [(m, v) for m, v in members if _cosine(v, centroid) >= self.s.consolidation_threshold]
            if len(core) >= 2:
                clusters.append(core)
        return clusters

    def consolidate(self, user_id: str, dry_run: bool = True) -> MaintenanceReport:
        before = self.store.count(user_id)
        report = MaintenanceReport(user_id=user_id, dry_run=dry_run, active_before=before, active_after=before)

        for cluster in self.find_clusters(user_id):
            mems = sorted((m for m, _ in cluster), key=lambda m: m.created_at)
            plan = ClusterPlan(source_ids=[m.id for m in mems], source_texts=[m.text for m in mems])
            report.clusters.append(plan)

            if len(mems) > self.s.max_cluster_size:
                plan.skipped_reason = "cluster too large; needs review"
                continue
            facts = self._merge(mems)
            if not facts:
                plan.skipped_reason = "merge failed; kept originals"
                continue
            if len(facts) >= len(mems):
                plan.skipped_reason = "nothing to merge"
                continue
            plan.merged = [f.text for f in facts]
            if dry_run:
                continue

            vectors = self.llm.embed([f.text for f in facts])
            new_ids = []
            for fact, vec in zip(facts, vectors):
                merged = Memory(
                    user_id=user_id, text=fact.text, category=fact.category,
                    importance=max([fact.importance, *(m.importance for m in mems)]),
                    access_count=sum(m.access_count for m in mems),
                    confirmation_count=sum(m.confirmation_count for m in mems),
                    source_ids=plan.source_ids, origin="consolidation")
                self.store.insert(merged, vec)
                new_ids.append(merged.id)
            for m in mems:  # soft delete; purge() removes them later
                self.store.invalidate(m.id, superseded_by=new_ids[0])
            plan.applied = True

        report.active_after = self.store.count(user_id)
        log.info("consolidation_done", extra={"ctx": {
            "user_id": user_id, "dry_run": dry_run, "clusters": len(report.clusters),
            "before": before, "after": report.active_after}})
        return report

    def purge(self, user_id: str, older_than_days: int | None = None) -> int:
        days = self.s.purge_after_days if older_than_days is None else older_than_days
        cutoff = utcnow() - timedelta(days=days)
        stale = [m.id for m, _ in self.store.iter_memories(user_id, active_only=False)
                 if not m.is_active and m.valid_until and m.valid_until <= cutoff]
        self.store.delete(stale)
        log.info("purge_done", extra={"ctx": {"user_id": user_id, "deleted": len(stale), "days": days}})
        return len(stale)

    def _merge(self, mems: list[Memory]) -> list[Fact]:
        listing = "\n".join(f'- "{m.text}" (created {m.created_at:%Y-%m-%d})' for m in mems)
        try:
            data = parse_json(self.llm.chat(
                [{"role": "system", "content": prompts.CONSOLIDATE_SYSTEM},
                 {"role": "user", "content": listing}], json_mode=True))
            return [Fact(**_coerce_fact(f)) for f in data.get("facts", []) if f.get("text")]
        except (LLMResponseError, ValueError, TypeError):
            log.warning("merge_unparseable", exc_info=True)
            return []
