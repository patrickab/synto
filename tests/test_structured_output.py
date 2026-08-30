"""
Tests for structured_output.py — the most critical module.
All tests use mocked OllamaClient; no Ollama required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from synto.models import AnalysisResult, CompilePlan, LintResult, PageSelection
from synto.ollama_client import OllamaClient
from synto.structured_output import (
    StructuredOutputError,
    _make_template,
    request_structured,
    request_text,
)


def _client(response: str) -> OllamaClient:
    c = MagicMock(spec=OllamaClient)
    c.generate.return_value = response
    return c


def _load_fixture(name: str) -> str:
    return (Path(__file__).parent / "fixtures" / name).read_text()


# ── Tier 1: direct JSON parse ──────────────────────────────────────────────────


def test_valid_analysis_json(fixtures_dir):
    raw = (fixtures_dir / "analysis_valid.json").read_text()
    result = request_structured(
        client=_client(raw),
        prompt="analyze",
        model_class=AnalysisResult,
        model="gemma4:e4b",
    )
    assert result.quality == "high"
    assert any(c.name == "quantum entanglement" for c in result.concepts)
    assert len(result.suggested_topics) > 0


def test_valid_compile_plan(fixtures_dir):
    raw = (fixtures_dir / "compile_plan_valid.json").read_text()
    result = request_structured(
        client=_client(raw),
        prompt="plan",
        model_class=CompilePlan,
        model="gemma4:e4b",
    )
    assert len(result.articles) == 1
    assert result.articles[0].action == "create"


def test_prose_wrapped_json_is_retried_not_salvaged(fixtures_dir):
    """Extraction heuristics are gone: a model that wraps JSON in prose gets retried."""
    inner = (fixtures_dir / "analysis_valid.json").read_text()
    wrapped = f"Sure, here you go:\n{inner}\nHope that helps!"
    with pytest.raises(StructuredOutputError):
        request_structured(
            client=_client(wrapped),
            prompt="analyze",
            model_class=AnalysisResult,
            model="gemma4:e4b",
            max_retries=0,
        )


# ── Tier 3: retry on failure ───────────────────────────────────────────────────


def test_retry_on_invalid_json(fixtures_dir):
    valid = (fixtures_dir / "analysis_valid.json").read_text()
    call_count = 0

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "not json at all"
        return valid

    c = MagicMock(spec=OllamaClient)
    c.generate.side_effect = lambda **kwargs: side_effect(**kwargs)

    result = request_structured(
        client=c,
        prompt="analyze",
        model_class=AnalysisResult,
        model="gemma4:e4b",
        max_retries=2,
    )
    assert result.quality == "high"
    assert call_count == 2  # failed once, succeeded on retry


def test_exhausted_retries_raises():
    c = _client("this is never valid json !!!!")
    with pytest.raises(StructuredOutputError):
        request_structured(
            client=c,
            prompt="analyze",
            model_class=AnalysisResult,
            model="gemma4:e4b",
            max_retries=1,
        )


def test_schema_validation_failure():
    # Valid JSON but wrong schema (missing required fields)
    bad = json.dumps({"wrong_field": "value"})
    c = _client(bad)
    with pytest.raises(StructuredOutputError):
        request_structured(
            client=c,
            prompt="analyze",
            model_class=AnalysisResult,
            model="gemma4:e4b",
            max_retries=0,
        )


def test_num_predict_passed_to_generate():
    """num_predict forwarded to client.generate so output isn't truncated mid-JSON."""
    c = _client(json.dumps({"pages": ["A"]}))
    request_structured(
        client=c,
        prompt="select",
        model_class=PageSelection,
        model="qwen2.5:14b",
        num_ctx=16384,
        num_predict=8192,
    )
    _, kwargs = c.generate.call_args
    assert kwargs.get("num_predict") == 8192


def test_num_predict_default_is_minus_one():
    """Default num_predict=-1 means unlimited — Ollama generates until stop token."""
    c = _client(json.dumps({"pages": ["A"]}))
    request_structured(
        client=c,
        prompt="select",
        model_class=PageSelection,
        model="qwen2.5:14b",
    )
    _, kwargs = c.generate.call_args
    assert kwargs.get("num_predict") == -1


def test_temperature_passed_to_generate():
    c = _client(json.dumps({"pages": ["A"]}))
    request_structured(
        client=c,
        prompt="select",
        model_class=PageSelection,
        model="qwen2.5:14b",
        temperature=0,
    )
    _, kwargs = c.generate.call_args
    assert kwargs.get("temperature") == 0


def test_json_schema_is_sent_as_format():
    """Ollama >=0.5 grammar-constrains on a schema; "json" only constrains syntax."""
    c = _client(json.dumps({"pages": ["A"]}))
    request_structured(
        client=c,
        prompt="select",
        model_class=PageSelection,
        model="qwen2.5:14b",
    )
    _, kwargs = c.generate.call_args
    assert kwargs["format"] == PageSelection.model_json_schema()
    assert "pages" in kwargs["format"]["properties"]


def test_truncated_json_fails_all_retries():
    """Truncated JSON (output cut off mid-string) exhausts retries and raises."""
    c = _client('{"pages": ["A')
    with pytest.raises(StructuredOutputError, match="Invalid JSON"):
        request_structured(
            client=c,
            prompt="select",
            model_class=PageSelection,
            model="qwen2.5:14b",
            max_retries=1,
        )


def test_missing_required_field_raises():
    c = _client(json.dumps({"wrong": []}))
    with pytest.raises(StructuredOutputError):
        request_structured(
            client=c,
            prompt="select",
            model_class=PageSelection,
            model="qwen2.5:14b",
            max_retries=0,
        )


# ── request_text: prose never touches the JSON path ───────────────────────────


def test_request_text_returns_body_verbatim():
    r"""The whole point: \nabla stays \nabla.

    Through JSON this string decodes to a newline plus a stranded "abla" —
    a *valid* escape, so there is no error to catch and no way to repair it.
    """
    body = "## Flow\n\n$$\\nabla p + \\nu \\Delta u = f$$\n\nWith $\\alpha \\in [0,1]$."
    result = request_text(client=_client(body), prompt="write", model="qwen2.5:14b")
    assert result == body
    assert result.count("\\nabla") == 1
    assert "\nabla" not in result


def test_request_text_sends_no_format():
    c = _client("body")
    request_text(client=c, prompt="write", model="m")
    _, kwargs = c.generate.call_args
    assert kwargs["format"] is None


# ── _make_template: nested object rendering ──────────────────────────────────


def test_template_expands_nested_object_array():
    """AnalysisResult.concepts is list[Concept] — template must show the object shape,
    not the array's description string."""
    template = json.loads(_make_template(AnalysisResult))
    concepts = template["concepts"]
    assert isinstance(concepts, list) and len(concepts) == 1
    assert isinstance(concepts[0], dict)
    assert set(concepts[0].keys()) == {"name", "aliases"}
    assert isinstance(concepts[0]["aliases"], list)


def test_template_does_not_cap_concept_count():
    """The rendered template must not surface a per-call concept ceiling.

    Why it matters: concepts are capped downstream by effective_max_concepts, from
    config. A number like "max 8" leaking from the schema into the template would tell the
    model to stop early and silently cap long-form sources below their configured ceiling —
    exactly the failure mode #52 fixed for multi-chunk sources, here for single-chunk ones.
    """
    template = _make_template(AnalysisResult)
    concepts = json.loads(template)["concepts"]
    # Object shape rendered (one example), with no numeric ceiling leaking from the schema.
    assert len(concepts) == 1 and set(concepts[0]) == {"name", "aliases"}
    assert "max 8" not in template.lower()


def test_template_expands_compile_plan_articles():
    template = json.loads(_make_template(CompilePlan))
    articles = template["articles"]
    assert isinstance(articles[0], dict)
    assert {"title", "action", "path", "reasoning", "source_paths"} <= set(articles[0].keys())
    assert set(articles[0]["action"].split(" | ")) == {"create", "update"}


def test_template_expands_lint_issues_with_enum():
    template = json.loads(_make_template(LintResult))
    issue = template["issues"][0]
    assert isinstance(issue, dict)
    assert "orphan" in issue["issue_type"]
    assert issue["auto_fixable"] is True


def test_template_primitive_array_keeps_description_hint():
    """list[str] still rendered as legacy single-string hint (not object)."""
    template = json.loads(_make_template(AnalysisResult))
    assert template["suggested_topics"] == [
        "Titles of wiki articles this note should feed into (max 5)"
    ]


def test_template_optional_field_keeps_outer_description():
    """Optional[str] (anyOf[str, null]) must still carry the parent field description."""
    template = json.loads(_make_template(AnalysisResult))
    assert "ISO 639-1" in template["language"]
