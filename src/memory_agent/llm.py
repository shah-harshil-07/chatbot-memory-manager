"""LLM access behind a tiny interface, so the rest of the code never imports OpenAI directly
(and tests can swap in a fake)."""
import json
import logging
import re
import time
from typing import Protocol

from openai import BadRequestError, OpenAI

from .config import Settings

log = logging.getLogger(__name__)


class LLMResponseError(RuntimeError):
    """The model returned something we could not parse or validate."""


class LLMClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def chat(self, messages: list[dict], *, json_mode: bool = False) -> str: ...


class OpenAIClient:
    def __init__(self, settings: Settings):
        self._s = settings
        # The SDK retries 408/409/429/5xx and connection errors with exponential backoff.
        self._client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_timeout_s,
            max_retries=settings.openai_max_retries,
        )
        self._json_mode_supported = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        kwargs: dict = {"model": self._s.openai_embedding_model, "input": texts}
        if self._s.openai_embedding_model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self._s.embedding_dim  # v3 models can shorten vectors
        start = time.perf_counter()
        resp = self._client.embeddings.create(**kwargs)
        log.debug("embeddings", extra={"ctx": {
            "n": len(texts), "tokens": resp.usage.total_tokens,
            "ms": round((time.perf_counter() - start) * 1000)}})
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    def chat(self, messages: list[dict], *, json_mode: bool = False) -> str:
        kwargs: dict = {"model": self._s.openai_chat_model, "messages": messages}
        if json_mode and self._json_mode_supported:
            kwargs["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            # Older models (e.g. gpt-4) reject JSON mode. The prompts already ask for JSON only
            # and parse_json tolerates fences, so retry without it and remember for next time.
            if "response_format" not in kwargs or getattr(exc, "param", None) != "response_format":
                raise
            log.warning("json_mode_unsupported", extra={"ctx": {"model": self._s.openai_chat_model}})
            self._json_mode_supported = False
            del kwargs["response_format"]
            resp = self._client.chat.completions.create(**kwargs)
        usage = resp.usage
        log.info("chat_completion", extra={"ctx": {
            "model": resp.model, "json": json_mode,
            "in_tokens": usage.prompt_tokens if usage else None,
            "out_tokens": usage.completion_tokens if usage else None,
            "ms": round((time.perf_counter() - start) * 1000)}})
        return resp.choices[0].message.content or ""


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"Invalid JSON from model: {text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise LLMResponseError("Expected a JSON object from model")
    return data
