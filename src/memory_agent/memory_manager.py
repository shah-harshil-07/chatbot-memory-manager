"""Write path: turn conversation into facts, then reconcile each fact with what is stored.

    message -> extract_facts() -> for each fact:
        similar = top-k existing memories
        nothing similar          -> ADD
        near-identical (>= 0.95) -> NOOP (confirm)        # no LLM call needed
        otherwise                -> ask LLM: ADD / UPDATE / DELETE / NOOP
"""
import logging

from pydantic import ValidationError

from . import prompts
from .config import Settings
from .llm import LLMClient, LLMResponseError, parse_json
from .models import Action, Category, Fact, Memory, MemoryOperation, ScoredMemory
from .store import MemoryStore

log = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, store: MemoryStore, llm: LLMClient, settings: Settings):
        self.store, self.llm, self.s = store, llm, settings

    # --------------------------------------------------------------- extract
    def extract_facts(self, user_message: str, assistant_reply: str | None = None) -> list[Fact]:
        turn = f"User: {user_message}"
        if assistant_reply:
            turn += f"\nAssistant: {assistant_reply}"
        try:
            data = parse_json(self.llm.chat(
                [{"role": "system", "content": prompts.EXTRACT_SYSTEM},
                 {"role": "user", "content": turn}], json_mode=True))
        except LLMResponseError:
            log.warning("fact_extraction_unparseable", exc_info=True)
            return []
        facts: list[Fact] = []
        for raw in data.get("facts", []):
            try:
                facts.append(Fact(**_coerce_fact(raw)))
            except (ValidationError, TypeError):
                log.warning("fact_invalid", extra={"ctx": {"raw": raw}})
        return facts

    # ------------------------------------------------------------- reconcile
    def remember(self, user_id: str, facts: list[Fact],
                 origin: str = "chat") -> list[MemoryOperation]:
        if not facts:
            return []
        vectors = self.llm.embed([f.text for f in facts])  # one batched call
        ops = [self._apply(user_id, fact, vec, origin) for fact, vec in zip(facts, vectors)]
        for op in ops:
            log.info("memory_op", extra={"ctx": {"user_id": user_id, **op.model_dump(mode="json")}})
        return ops

    def process_turn(self, user_id: str, user_message: str,
                     assistant_reply: str | None = None) -> list[MemoryOperation]:
        return self.remember(user_id, self.extract_facts(user_message, assistant_reply))

    def remember_text(self, user_id: str, text: str) -> list[MemoryOperation]:
        """Explicit 'remember this' — still extracted + reconciled, so no duplicates/contradictions."""
        facts = self.extract_facts(f"Please remember: {text}") or [Fact(text=text)]
        return self.remember(user_id, facts, origin="explicit")

    def _apply(self, user_id: str, fact: Fact, vector: list[float], origin: str) -> MemoryOperation:
        similar = self.store.search(user_id, vector, limit=self.s.reconcile_top_k)

        if not similar:
            return self._add(user_id, fact, vector, origin, reason="no related memories")

        top = similar[0]
        if top.score >= self.s.duplicate_threshold:
            self.store.confirm(top.memory)
            return MemoryOperation(action=Action.NOOP, fact=fact.text, memory_id=top.memory.id,
                                   reason=f"near-duplicate (score {top.score:.2f})")

        decision = self._decide(fact, similar)
        action, target = decision["action"], decision["target"]

        if action is Action.NOOP and target:
            self.store.confirm(target)
            return MemoryOperation(action=action, fact=fact.text, memory_id=target.id, reason=decision["reason"])

        if action is Action.UPDATE and target:
            new_fact = fact.model_copy(update={"text": decision["text"] or fact.text})
            new_vec = vector if new_fact.text == fact.text else self.llm.embed([new_fact.text])[0]
            new = Memory(user_id=user_id, text=new_fact.text, category=new_fact.category,
                         importance=max(new_fact.importance, target.importance),
                         confirmation_count=target.confirmation_count,
                         source_ids=[target.id, *target.source_ids], origin=origin)  # type: ignore[arg-type]
            self.store.insert(new, new_vec)
            self.store.invalidate(target.id, superseded_by=new.id)
            return MemoryOperation(action=action, fact=new.text, memory_id=new.id,
                                   target_id=target.id, reason=decision["reason"])

        if action is Action.DELETE and target:
            self.store.invalidate(target.id)
            return MemoryOperation(action=action, fact=fact.text, target_id=target.id, reason=decision["reason"])

        # ADD, or any decision we couldn't act on safely: keep the information.
        return self._add(user_id, fact, vector, origin, reason=decision["reason"])

    def _add(self, user_id: str, fact: Fact, vector: list[float], origin: str, reason: str) -> MemoryOperation:
        mem = Memory(user_id=user_id, text=fact.text, category=fact.category,
                     importance=fact.importance, origin=origin)  # type: ignore[arg-type]
        self.store.insert(mem, vector)
        return MemoryOperation(action=Action.ADD, fact=fact.text, memory_id=mem.id, reason=reason)

    def _decide(self, fact: Fact, similar: list[ScoredMemory]) -> dict:
        # Show the model short aliases ("m1") instead of UUIDs: fewer tokens, no hallucinated ids.
        aliases = {f"m{i}": s.memory for i, s in enumerate(similar, 1)}
        existing = "\n".join(f'{a}: "{m.text}" (stored {m.created_at:%Y-%m-%d})' for a, m in aliases.items())
        fallback = {"action": Action.ADD, "target": None, "text": None, "reason": "fallback: unparseable decision"}
        try:
            data = parse_json(self.llm.chat(
                [{"role": "system", "content": prompts.RECONCILE_SYSTEM},
                 {"role": "user", "content": prompts.RECONCILE_USER.format(existing=existing, fact=fact.text)}],
                json_mode=True))
            action = Action(str(data.get("action", "")).upper())
        except (LLMResponseError, ValueError):
            log.warning("reconcile_unparseable", exc_info=True)
            return fallback
        target = aliases.get(str(data.get("target_id") or ""))
        if action in (Action.UPDATE, Action.DELETE, Action.NOOP) and target is None:
            log.warning("reconcile_missing_target", extra={"ctx": {"data": data}})
            return fallback
        return {"action": action, "target": target, "text": data.get("text"),
                "reason": str(data.get("reason") or "")[:300]}

def _coerce_fact(raw: dict) -> dict:
    """Be lenient with model output: unknown categories -> other, clamp importance."""
    category = str(raw.get("category", "other")).lower()
    try:
        importance = int(raw.get("importance", 5))
    except (TypeError, ValueError):
        importance = 5
    return {
        "text": str(raw.get("text", "")).strip(),
        "category": category if category in Category._value2member_map_ else "other",
        "importance": min(10, max(1, importance)),
    }
