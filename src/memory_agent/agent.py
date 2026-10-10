"""The agent loop — same 4 steps as the tiny demo:

    1. RECALL  embed the message, search the user's active memories
    2. PROMPT  put recalled memories into the system prompt (+ recent chat history)
    3. ANSWER  call the chat model
    4. WRITE   extract facts from the turn and reconcile them into memory
"""
import logging
from datetime import datetime, timezone

from . import prompts
from .config import Settings
from .llm import LLMClient
from .memory_manager import MemoryManager
from .models import ChatMessage, ScoredMemory, TurnResult
from .store import MemoryStore

log = logging.getLogger(__name__)


class MemoryAgent:
    def __init__(self, store: MemoryStore, llm: LLMClient, manager: MemoryManager, settings: Settings):
        self.store, self.llm, self.manager, self.s = store, llm, manager, settings

    # Step 1
    def recall(self, user_id: str, query: str) -> list[ScoredMemory]:
        vector = self.llm.embed([query])[0]
        hits = self.store.search(user_id, vector, limit=self.s.recall_top_k, min_score=self.s.recall_min_score)
        self.store.touch([h.memory for h in hits])
        return hits

    # Step 2
    def build_messages(self, recalled: list[ScoredMemory], message: str,
                       history: list[ChatMessage]) -> list[dict]:
        memory_lines = "\n".join(f"- {h.memory.text}" for h in recalled) or "- (nothing yet)"
        system = prompts.CHAT_SYSTEM.format(
            today=datetime.now(timezone.utc).strftime("%Y-%m-%d"), memories=memory_lines)
        recent = history[-self.s.history_turns:] if self.s.history_turns else []
        return [{"role": "system", "content": system},
                *({"role": m.role, "content": m.content} for m in recent),
                {"role": "user", "content": message}]

    # Steps 1-3 (what the user waits for)
    def respond(self, user_id: str, message: str,
                history: list[ChatMessage] | None = None) -> tuple[str, list[ScoredMemory]]:
        recalled = self.recall(user_id, message)
        reply = self.llm.chat(self.build_messages(recalled, message, history or []))
        log.info("turn_answered", extra={"ctx": {"user_id": user_id, "recalled": len(recalled)}})
        return reply, recalled

    # Step 4 (can run in the background, after the reply is sent)
    def write(self, user_id: str, message: str, reply: str):
        return self.manager.process_turn(user_id, message, reply)

    def run_turn(self, user_id: str, message: str,
                 history: list[ChatMessage] | None = None, write: bool = True) -> TurnResult:
        reply, recalled = self.respond(user_id, message, history)
        ops = self.write(user_id, message, reply) if write else []
        return TurnResult(reply=reply, recalled=recalled, operations=ops)
