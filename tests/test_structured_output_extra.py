"""Additional tests for structured_output.py uncovered paths."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from synto.models import PageSelection
from synto.ollama_client import OllamaClient
from synto.structured_output import (
    _render_example,
    _try_parse,
    request_structured,
)


def _client(response: str) -> OllamaClient:
    c = MagicMock(spec=OllamaClient)
    c.generate.return_value = response
    return c


# ── _try_parse ───────────────────────────────────────────────────────────────


def test_try_parse_invalid_json_returns_error():
    """Invalid JSON returns (None, error_string)."""
    result, error = _try_parse("not json", PageSelection)
    assert result is None
    assert error


def test_try_parse_valid_json_wrong_schema():
    """Valid JSON but wrong schema returns (None, error_string)."""
    raw = json.dumps({"wrong": "schema"})
    result, error = _try_parse(raw, PageSelection)
    assert result is None
    assert error


def test_try_parse_does_not_unwrap_containers():
    """A {"ClassName": {...}} wrapper is a failure, not something to guess around.

    Unwrapping was a small-model accommodation; the schema is now sent to the
    provider, so a wrapped response means the request went wrong and should retry.
    """
    raw = json.dumps({"PageSelection": {"pages": ["A"]}})
    result, error = _try_parse(raw, PageSelection)
    assert result is None
    assert error


# ── _render_example ──────────────────────────────────────────────────────────


def test_render_example_anyof_with_null():
    """anyOf with null alternative skips to non-null."""
    schema_node = {
        "anyOf": [{"type": "null"}, {"type": "string", "description": "A string"}],
        "description": "Optional field",
    }
    result = _render_example(schema_node, {})
    assert result == "A string"


def test_render_example_anyof_all_null():
    """anyOf with only null → returns None."""
    schema_node = {"anyOf": [{"type": "null"}]}
    result = _render_example(schema_node, {})
    assert result is None


# ── request_structured with retry error feedback ─────────────────────────────


def test_retry_includes_error_feedback():
    """Retry attempts include the previous error in the prompt."""
    call_count = 0
    captured_prompts = []

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        captured_prompts.append(kwargs.get("prompt", ""))
        if call_count == 1:
            return "bad json"
        return json.dumps({"pages": ["A"]})

    c = MagicMock(spec=OllamaClient)
    c.generate.side_effect = side_effect

    result = request_structured(
        client=c,
        prompt="select pages",
        model_class=PageSelection,
        model="test",
        max_retries=1,
    )
    assert result.pages == ["A"]
    assert call_count == 2
    # Second prompt should mention the error
    assert "error" in captured_prompts[1].lower() or "invalid" in captured_prompts[1].lower()
