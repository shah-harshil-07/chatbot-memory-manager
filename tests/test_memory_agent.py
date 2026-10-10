import json
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from memory_agent.api import create_app
from memory_agent.llm import LLMResponseError, parse_json
from memory_agent.models import Action, Fact, utcnow


# ------------------------------------------------------------------- store
def test_store_isolates_users_and_hides_invalidated(container, fake_llm):
    mm = container.manager
    mm.remember("alice", [Fact(text="User likes green tea")])
    mm.remember("bob", [Fact(text="User likes black coffee")])
    vec = fake_llm.embed(["likes tea"])[0]

    hits = container.store.search("alice", vec, limit=5)
    assert [h.memory.text for h in hits] == ["User likes green tea"]
    assert container.store.search("bob", vec, limit=5)[0].memory.user_id == "bob"

    container.store.invalidate(hits[0].memory.id)
    assert container.store.search("alice", vec, limit=5) == []
    assert container.store.count("alice", active_only=False) == 1  # kept as history


# ---------------------------------------------------------- reconciliation
def test_add_then_duplicate_is_noop(container):
    ops1 = container.manager.process_turn("u1", "I am on the Finance team")
    ops2 = container.manager.process_turn("u1", "I am on the Finance team")
    assert [o.action for o in ops1] == [Action.ADD]
    assert [o.action for o in ops2] == [Action.NOOP]
    [mem] = container.store.list("u1")
    assert mem.confirmation_count == 2


def test_contradiction_updates_and_keeps_history(container):
    container.manager.process_turn("u1", "I am on the Finance team")
    [op] = container.manager.process_turn("u1", "I moved to the Data Platform team")

    assert op.action is Action.UPDATE
    active = container.store.list("u1")
    assert [m.text for m in active] == ["User is on the Data Platform team"]
    old = container.store.get(op.target_id)
    assert old and not old.is_active and old.superseded_by == op.memory_id
    assert active[0].source_ids == [old.id]


def test_delete_action_invalidates(container, fake_llm):
    container.manager.remember("u1", [Fact(text="User owns a red car")])
    fake_llm.handlers["reconcile"] = lambda m: json.dumps(
        {"action": "DELETE", "target_id": "m1", "text": None, "reason": "sold car"})
    [op] = container.manager.remember("u1", [Fact(text="User no longer owns a car")])
    assert op.action is Action.DELETE
    assert container.store.list("u1") == []


def test_unparseable_decision_falls_back_to_add(container, fake_llm):
    container.manager.remember("u1", [Fact(text="User likes hiking trails")])
    fake_llm.handlers["reconcile"] = lambda m: "not json at all"
    [op] = container.manager.remember("u1", [Fact(text="User likes mountain hiking")])
    assert op.action is Action.ADD
    assert container.store.count("u1") == 2


def test_hallucinated_target_falls_back_to_add(container, fake_llm):
    container.manager.remember("u1", [Fact(text="User likes hiking trails")])
    fake_llm.handlers["reconcile"] = lambda m: json.dumps({"action": "UPDATE", "target_id": "m99", "text": "x"})
    [op] = container.manager.remember("u1", [Fact(text="User likes mountain hiking")])
    assert op.action is Action.ADD


# -------------------------------------------------------------------- agent
def test_turn_recalls_memory_into_prompt(container, fake_llm):
    container.agent.run_turn("u1", "Hi, my name is Priya")
    container.agent.run_turn("u1", "I am on the Finance team")
    container.agent.run_turn("u1", "I moved to the Data Platform team")
    result = container.agent.run_turn("u1", "Which team am I on?")

    assert "Data Platform" in result.reply and "Finance" not in result.reply
    assert result.operations == []  # questions add nothing
    chat_prompts = [m for kind, m in fake_llm.calls if kind == "chat"]
    assert "User is on the Data Platform team" in chat_prompts[-1][0]["content"]


# -------------------------------------------------------------- maintenance
def _seed_duplicates(container):
    container.manager.remember("u1", [Fact(text="User is on the Finance team")])
    container.manager.remember("u1", [Fact(text="User prefers dark mode")])
    container.manager.remember("u1", [Fact(text="User prefers dark mode everywhere")])
    container.manager.remember("u1", [Fact(text="User really prefers dark mode")])


def test_consolidate_dry_run_changes_nothing(container):
    _seed_duplicates(container)
    report = container.maintenance.consolidate("u1", dry_run=True)
    assert len(report.clusters) == 1 and report.clusters[0].merged
    assert container.store.count("u1") == 4


def test_consolidate_merges_cluster_with_provenance(container):
    _seed_duplicates(container)
    report = container.maintenance.consolidate("u1", dry_run=False)
    assert report.active_before == 4 and report.active_after == 2
    merged = next(m for m in container.store.list("u1") if m.origin == "consolidation")
    assert len(merged.source_ids) == 3
    assert merged.confirmation_count == 3


def test_purge_only_removes_old_invalidated(container):
    container.manager.remember("u1", [Fact(text="User lives in Pune")])
    [mem] = container.store.list("u1")
    container.store.invalidate(mem.id)
    assert container.maintenance.purge("u1") == 0  # too recent
    container.store.set_fields(mem.id, valid_until=utcnow() - timedelta(days=60))
    assert container.maintenance.purge("u1") == 1
    assert container.store.count("u1", active_only=False) == 0


# ---------------------------------------------------------------------- api
@pytest.fixture
def client(container):
    with TestClient(create_app(container)) as c:
        yield c


def test_api_chat_and_memories(client):
    r = client.post("/v1/users/u1/chat", json={"message": "I am on the Finance team", "sync_write": True})
    assert r.status_code == 200 and r.json()["operations"][0]["action"] == "ADD"

    mems = client.get("/v1/users/u1/memories").json()
    assert [m["text"] for m in mems] == ["User is on the Finance team"]

    assert client.delete(f"/v1/users/someone-else/memories/{mems[0]['id']}").status_code == 404
    assert client.delete(f"/v1/users/u1/memories/{mems[0]['id']}").status_code == 204
    assert client.get("/v1/users/u1/memories").json() == []
    assert client.get("/health").json()["qdrant"] is True


def test_api_background_write(client):
    r = client.post("/v1/users/u2/chat", json={"message": "my name is Asha"})
    assert r.json()["operations"] is None
    # TestClient runs background tasks before returning
    assert client.get("/v1/users/u2/memories").json()[0]["text"] == "User's name is Asha"


def test_api_key_enforced(container):
    from pydantic import SecretStr
    container.settings.app_api_key = SecretStr("secret")
    with TestClient(create_app(container)) as c:
        assert c.get("/v1/users/u1/memories").status_code == 401
        assert c.get("/v1/users/u1/memories", headers={"X-API-Key": "secret"}).status_code == 200


# --------------------------------------------------------------------- misc
def test_parse_json_handles_fences_and_errors():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(LLMResponseError):
        parse_json("nope")
