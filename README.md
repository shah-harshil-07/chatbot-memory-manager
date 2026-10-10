# Memory Agent

A conversational agent with **long-term memory**, built on **OpenAI** (chat + embeddings) and **Qdrant** (vector store).
It is the production version of the tiny SQLite demo: same four-step loop, real components.

```
user message
   │
   ▼
1. RECALL   embed message → Qdrant search (this user, active memories only)
2. PROMPT   recalled facts go into the system prompt (+ recent chat history)
3. ANSWER   OpenAI chat model replies            ← user gets the reply here
4. WRITE    (background) extract facts → reconcile each: ADD / UPDATE / DELETE / NOOP
```

Maintenance jobs keep memory small over time: **consolidate** near-duplicates and **purge** old invalidated memories.

---

## Quick start

### 1. Configure
```bash
cp .env.example .env
# set OPENAI_API_KEY and OPENAI_CHAT_MODEL (and EMBEDDING_DIM if you change the embedding model)
```

### 2a. Run everything with Docker
```bash
docker compose up --build
# API:    http://localhost:8000/docs
# Qdrant: http://localhost:6333/dashboard   (browse the stored memories)
```

### 2b. Or run locally (Qdrant in Docker, app in a virtualenv)
```bash
docker compose up -d qdrant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
memory-agent chat --user priya --show-memory
```

---

## Using it

### CLI
```bash
memory-agent chat --user priya --show-memory     # chat; shows recalled memories and ADD/UPDATE/... per turn
memory-agent memories --user priya [--all]       # list memories (--all includes invalidated history)
memory-agent remember --user priya "I prefer short answers"
memory-agent forget --user priya <memory-id>     # hard delete
memory-agent consolidate --user priya [--apply]  # dry run by default
memory-agent purge --user priya [--days 30]
memory-agent serve --port 8000
```

### HTTP API
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + Qdrant connectivity |
| POST | `/v1/users/{user_id}/chat` | Chat. Body: `message`, optional `history`, `sync_write` |
| GET | `/v1/users/{user_id}/memories?include_inactive=false` | List memories |
| POST | `/v1/users/{user_id}/memories` | Explicitly remember `text` (still reconciled) |
| DELETE | `/v1/users/{user_id}/memories/{id}` | Hard delete one memory |
| POST | `/v1/users/{user_id}/maintenance/consolidate?dry_run=true` | Merge near-duplicates |
| POST | `/v1/users/{user_id}/maintenance/purge?older_than_days=30` | Delete old invalidated memories |

```bash
curl -s localhost:8000/v1/users/priya/chat -H 'Content-Type: application/json' \
  -d '{"message": "I moved to the Data Platform team", "sync_write": true}' | jq
```
If `APP_API_KEY` is set, send it as the `X-API-Key` header.

The server is **stateless**: the client sends recent `history`; long-term memory lives in Qdrant.
By default, memory writes run **after** the reply (FastAPI background task), so users never wait for them.
`sync_write: true` waits and returns the operations, which is handy for debugging.

---

## Project layout

```
src/memory_agent/
├── config.py          Settings from env/.env (pydantic-settings)
├── logging_config.py  Text or JSON logs
├── models.py          Memory, Fact, MemoryOperation, reports (pydantic)
├── llm.py             OpenAI wrapper behind a small LLMClient protocol (+ JSON parsing)
├── store.py           Qdrant store: insert/search/invalidate/touch/delete/scroll
├── prompts.py         Every prompt, in one reviewable file
├── memory_manager.py  Write path: extract facts → reconcile (ADD/UPDATE/DELETE/NOOP)
├── agent.py           The 4-step loop
├── maintenance.py     Consolidation (ANN neighbours → union-find → LLM merge) + purge
├── container.py       Wires components together
├── api.py             FastAPI service
└── cli.py             Command-line interface
tests/                 Fake LLM + in-memory Qdrant; no API key or Docker needed
```

## How a memory is stored

One Qdrant **point** per memory: `id` = UUID, `vector` = embedding of the text, `payload` = everything else:

```json
{
  "user_id": "priya",
  "text": "User is on the Data Platform team",
  "category": "work", "importance": 7,
  "created_at": "...", "updated_at": "...", "last_accessed_at": "...",
  "access_count": 3, "confirmation_count": 2,
  "is_active": true, "valid_until": null, "superseded_by": null,
  "source_ids": ["<id of the old 'Finance team' memory>"],
  "origin": "chat"
}
```

Payload indexes on `user_id` and `is_active` keep the per-user, active-only filter fast as the collection grows.

## Design decisions

- **Invalidate, don't delete.** Contradicted or merged memories get `is_active=false`, `valid_until`, `superseded_by`.
  Recall never sees them; history and provenance are preserved. `purge` hard-deletes them after `PURGE_AFTER_DAYS`.
- **Cheap path first.** A near-identical fact (similarity ≥ `DUPLICATE_THRESHOLD`) is a NOOP without an LLM call;
  a fact with no related memories is an ADD without an LLM call. The LLM decides only the ambiguous cases.
- **Safe fallbacks.** If the model returns invalid JSON or references an unknown memory, the fact is ADDed rather than
  dropped, and nothing existing is changed. Consolidation that fails keeps the originals.
- **Short aliases for the LLM.** Reconciliation shows memories as `m1`, `m2`, … instead of UUIDs: fewer tokens, no invented ids.
- **Chaining guard.** Consolidation clusters are trimmed to members close to the cluster centroid, and oversized clusters are skipped.
- **Retries.** The OpenAI SDK retries 429/5xx/timeouts with exponential backoff (`OPENAI_MAX_RETRIES`). The API maps
  provider rate limits to `503 + Retry-After` and other provider errors to `502`. A failed background write is logged, never surfaced to the user.
- **Secrets.** Never extracted into memory (prompt rule), never logged (`SecretStr`).

## Configuration

All settings are in `config.py`; see `.env.example`. The most important ones:

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | required | Your key |
| `OPENAI_CHAT_MODEL` | required | Chat model used for answers, extraction, reconciliation, consolidation |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `EMBEDDING_DIM` | `1536` | Vector size; must match the model (v3 models can be shortened) |
| `RECALL_TOP_K` / `RECALL_MIN_SCORE` | `5` / `0.30` | How many memories enter the prompt, and the relevance cut-off |
| `DUPLICATE_THRESHOLD` | `0.95` | Similarity treated as "same fact" |
| `CONSOLIDATION_THRESHOLD` | `0.88` | Similarity treated as "near-duplicate" during maintenance |
| `PURGE_AFTER_DAYS` | `30` | Grace period before invalidated memories are deleted |

Changing the embedding model or dimension needs a **new collection** (`QDRANT_COLLECTION`); the app refuses to start on a mismatch.

## Tests

```bash
pytest -q
```
The suite covers the store (user isolation, invalidation), every reconciliation path including fallbacks,
the agent loop, consolidation (dry run and apply, with provenance), purge, and the API (sync and background writes, ownership checks, API key).

## Known limits and next steps

- Consolidation runs one ANN query per memory. That's fine for thousands of memories per user; for very large users,
  use `query_batch_points` and run it as a scheduled job (cron / Celery / Kubernetes CronJob) rather than via the API.
- Counters (`access_count`, `confirmation_count`) use read-modify-write; small races only affect statistics.
- Add per-user rate limiting at your gateway, and an evaluation set ("must-not-forget" questions) before changing prompts or thresholds.
