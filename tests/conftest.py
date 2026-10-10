"""Fixtures: a deterministic fake LLM + Qdrant's in-process ':memory:' mode.
No API key, no network, no Docker needed to run the test suite."""
import hashlib
import json
import math
import re
from collections.abc import Callable

import pytest
from qdrant_client import QdrantClient

from memory_agent import prompts
from memory_agent.config import Settings
from memory_agent.container import build_container

DIM = 256
STOP = {"i", "am", "is", "the", "a", "an", "on", "my", "what", "which", "user", "to", "of", "in"}


class FakeLLM:
    """Hash-based embeddings + rule-based 'chat' that recognises our prompts.
    Override any step per test via the `handlers` dict."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict]]] = []
        self.handlers: dict[str, Callable[[list[dict]], str]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * DIM
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if w not in STOP:
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

    def chat(self, messages: list[dict], *, json_mode: bool = False) -> str:
        system = messages[0]["content"]
        kind = ("extract" if system == prompts.EXTRACT_SYSTEM else
                "reconcile" if system == prompts.RECONCILE_SYSTEM else
                "consolidate" if system == prompts.CONSOLIDATE_SYSTEM else "chat")
        self.calls.append((kind, messages))
        if kind in self.handlers:
            return self.handlers[kind](messages)
        return getattr(self, f"_{kind}")(messages)

    # ---- default behaviours
    @staticmethod
    def _extract(messages):
        text = messages[-1]["content"].split("\nAssistant:")[0].lower()
        facts = []
        if m := re.search(r"my name is (\w+)", text):
            facts.append({"text": f"User's name is {m[1].title()}", "category": "profile", "importance": 8})
        if m := (re.search(r"(?:i'm|i am) on the ([\w ]+?) team", text)
                 or re.search(r"i moved to the ([\w ]+?) team", text)):
            facts.append({"text": f"User is on the {m[1].title()} team", "category": "work", "importance": 7})
        return json.dumps({"facts": facts})

    @staticmethod
    def _reconcile(messages):
        body = messages[-1]["content"]
        new = body.split("NEW fact: ")[1].strip()
        if "team" in new:
            for line in body.splitlines():
                if line.startswith("m") and "team" in line:
                    return json.dumps({"action": "UPDATE", "target_id": line.split(":")[0],
                                       "text": new, "reason": "team changed"})
        return json.dumps({"action": "ADD", "target_id": None, "text": new, "reason": "new"})

    @staticmethod
    def _consolidate(messages):
        first = re.search(r'"(.+?)"', messages[-1]["content"])[1]
        return json.dumps({"facts": [{"text": first, "category": "preference", "importance": 5}]})

    @staticmethod
    def _chat(messages):
        return "ANSWER using: " + messages[0]["content"].split("(most relevant first; may be incomplete):\n")[1].split("\n\n")[0]


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, openai_api_key="test", openai_chat_model="fake",  # type: ignore[call-arg]
                    embedding_dim=DIM, recall_min_score=0.1, consolidation_threshold=0.8)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def container(settings, fake_llm):
    return build_container(settings, llm=fake_llm, qdrant=QdrantClient(":memory:"))
