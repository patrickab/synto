"""
Talking to an LLM: plain text, or a small JSON object.

  request_text       — markdown, prose, LaTeX. No envelope, nothing to unescape.
  request_structured — label-shaped data only (names, enums, numbers, string
                       lists), grammar-constrained to the model's JSON Schema
                       and validated with Pydantic. Retry on failure; never
                       guess at a malformed response.

Prose does not go through request_structured. JSON's escape character is
LaTeX's command character, and ``\n`` is a *valid* JSON escape — so ``\nabla``
silently decodes to a newline plus a stranded ``abla``, with no parse error to
catch and no way to tell it from a real paragraph break afterwards.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ValidationError

from .metrics import LLMCallEvent, emit

if TYPE_CHECKING:
    from .protocols import LLMClientProtocol

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)

# Template-based instruction: a concrete fill-in example is less likely to
# confuse small models than a full JSON Schema object (which they may echo back).
_SCHEMA_INSTRUCTION = """\
You MUST respond with ONLY valid JSON. No prose before or after.
Return the JSON object directly. Do NOT wrap it or add extra keys.

Fill in this exact JSON structure with real content:

{template}

Replace each placeholder string with actual content. Keep the same keys and types.
Respond with nothing but the completed JSON object."""


class StructuredOutputError(Exception):
    pass


def _resolve_ref(node: dict, defs: dict) -> dict:
    if "$ref" in node:
        key = node["$ref"].rsplit("/", 1)[-1]
        resolved = defs.get(key)
        return resolved if isinstance(resolved, dict) else node
    return node


def _render_example(node: dict, defs: dict, field_name: str = "") -> object:
    """Render a schema node as a fill-in JSON example value."""
    node = _resolve_ref(node, defs)

    if "anyOf" in node and "type" not in node:
        outer_desc = node.get("description", "")
        for alt in node["anyOf"]:
            if alt.get("type") != "null":
                if outer_desc and "description" not in alt:
                    alt = {**alt, "description": outer_desc}
                return _render_example(alt, defs, field_name)
        return None

    ftype = node.get("type")
    desc = node.get("description", "")
    enum = node.get("enum")

    if enum:
        return " | ".join(str(e) for e in enum)
    if ftype == "array":
        items = _resolve_ref(node.get("items", {}), defs)
        if items.get("type") == "object" or "properties" in items:
            return [_render_example(items, defs, field_name)]
        return [desc[:60] or f"<{field_name} item>"]
    if ftype == "object" or "properties" in node:
        return {
            sub_name: _render_example(sub, defs, sub_name)
            for sub_name, sub in node.get("properties", {}).items()
        }
    if ftype in ("integer", "number"):
        return 0
    if ftype == "boolean":
        return True
    return desc[:80] or f"<{field_name}>"


def _make_template(model_class: type[T]) -> str:
    """Build a fill-in JSON example from model fields (simpler than raw JSON Schema).

    Recursively expands nested objects so small models see the real structure
    for fields typed as `list[NestedModel]` instead of the array description.
    """
    schema = model_class.model_json_schema()
    defs = schema.get("$defs", {}) or schema.get("definitions", {})
    props = schema.get("properties", {})
    template = {name: _render_example(sub, defs, name) for name, sub in props.items()}
    return json.dumps(template, indent=2)


def _schema_system(model_class: type[T]) -> str:
    template = _make_template(model_class)
    return _SCHEMA_INSTRUCTION.format(template=template)


def _try_parse(raw: str, model_class: type[T]) -> tuple[T | None, str]:
    """Parse JSON + validate against the model. Returns (result, error_str).

    No escape repair, no container unwrapping: this path only ever carries
    label-shaped data, so a failure here is a real failure and belongs in the
    retry loop rather than being guessed at.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    try:
        return model_class.model_validate(data), ""
    except ValidationError as e:
        return None, str(e)


def request_text(
    client: LLMClientProtocol,
    prompt: str,
    model: str,
    system: str = "",
    num_ctx: int = 8192,
    num_predict: int = -1,
    temperature: float | None = None,
    stage: str = "",
    model_role: str | None = None,
    think: bool | None = None,
    options: dict | None = None,
) -> str:
    """Request plain-text output — no JSON envelope, no parsing.

    Prose and LaTeX must never travel inside a JSON string: ``\\n`` is a valid JSON
    escape, so ``\\nabla`` decodes to a newline plus a stranded ``abla`` with no
    error to catch. Use this for any field that carries markdown or math; keep
    ``request_structured`` for label-shaped data (names, enums, numbers).
    """
    raw = client.generate(
        prompt=prompt,
        model=model,
        system=system,
        format=None,
        num_ctx=num_ctx,
        num_predict=num_predict,
        temperature=temperature,
        think=think,
        options=options,
    )
    stats = getattr(client, "_last_stats", {}) or {}
    emit(
        LLMCallEvent(
            stage=stage,
            model=model,
            tier=1,
            retries=0,
            latency_ms=int(stats.get("latency_ms") or 0),
            prompt_tokens=stats.get("prompt_tokens"),
            completion_tokens=stats.get("completion_tokens"),
            num_ctx=num_ctx,
            error=None,
            model_role=model_role,
        )
    )
    return raw.strip()


def request_structured(
    client: LLMClientProtocol,
    prompt: str,
    model_class: type[T],
    model: str,
    system: str = "",
    num_ctx: int = 8192,
    num_predict: int = -1,
    temperature: float | None = None,
    max_retries: int = 2,
    stage: str = "",
    model_role: str | None = None,
    think: bool | None = None,
    options: dict | None = None,
) -> T:
    """
    Request structured output from an LLM client, parse into Pydantic model.

    Args:
        client:      LLM client (OllamaClient or OpenAICompatClient)
        prompt:      User-facing prompt
        model_class: Pydantic model to parse response into
        model:       Model name (passed to the LLM client)
        system:      Optional domain context (prepended before schema instruction)
        num_ctx:     Context window size
        max_retries: How many retry attempts after initial failure
        stage:       Pipeline stage tag for metrics ("ingest", "compile_article", etc.)

    Raises:
        StructuredOutputError: if all attempts exhausted
    """
    schema_system = _schema_system(model_class)
    full_system = f"{system}\n\n{schema_system}" if system.strip() else schema_system

    last_error: str = ""
    current_prompt = prompt

    total_latency_ms = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    prompt_tokens_seen = False
    completion_tokens_seen = False

    def _emit(tier: int, retries: int, error: str | None) -> None:
        emit(
            LLMCallEvent(
                stage=stage,
                model=model,
                tier=tier,
                retries=retries,
                latency_ms=total_latency_ms,
                prompt_tokens=total_prompt_tokens if prompt_tokens_seen else None,
                completion_tokens=total_completion_tokens if completion_tokens_seen else None,
                num_ctx=num_ctx,
                error=error,
                model_role=model_role,
            )
        )

    for attempt in range(max_retries + 1):
        log.debug("structured_output attempt %d/%d model=%s", attempt + 1, max_retries + 1, model)

        # Ollama >=0.5 grammar-constrains decoding to the schema, so output cannot
        # be malformed or the wrong shape. Providers that only understand "json"
        # (OpenAI-compat json_object mode) fall back to syntax-only constraint and
        # lean on Pydantic validation + retry below.
        raw = client.generate(
            prompt=current_prompt,
            model=model,
            system=full_system,
            format=model_class.model_json_schema(),
            num_ctx=num_ctx,
            num_predict=num_predict,
            temperature=temperature,
            think=think,
            options=options,
        )
        stats = getattr(client, "_last_stats", {}) or {}
        total_latency_ms += int(stats.get("latency_ms") or 0)
        pt = stats.get("prompt_tokens")
        ct = stats.get("completion_tokens")
        if pt is not None:
            total_prompt_tokens += int(pt)
            prompt_tokens_seen = True
        if ct is not None:
            total_completion_tokens += int(ct)
            completion_tokens_seen = True

        result, parse_err = _try_parse(raw, model_class)
        if result is not None:
            _emit(tier=3 if attempt > 0 else 1, retries=attempt, error=None)
            return result
        last_error = parse_err

        log.debug(
            "structured_output attempt %d failed: %s. Raw (first 300): %s",
            attempt + 1,
            last_error,
            raw[:300],
        )

        if attempt < max_retries:
            current_prompt = (
                f"Your previous response was invalid.\n"
                f"Error: {last_error}\n\n"
                f"Original request:\n{prompt}\n\n"
                f"Respond with ONLY valid JSON matching the schema. Nothing else."
            )

    _emit(tier=-1, retries=max_retries, error=last_error)
    raise StructuredOutputError(
        f"Failed to get valid {model_class.__name__} after {max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
