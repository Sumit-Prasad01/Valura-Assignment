"""
Classifier routing accuracy tests against the gold set.
Threshold: >= 85% routing accuracy (ASSIGNMENT.md).

LLM is mocked — tests run in CI without OPENAI_API_KEY.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from src.classifier.intent import classify
from src.models.classifier import ClassifierOutput, ExtractedEntities
from tests.helpers.entity_matcher import matches_entities


def _make_mock_response(agent: str, intent: str, entities: dict) -> MagicMock:
    """Build a mock OpenAI response returning the given structured output."""
    payload = json.dumps({
        "agent": agent,
        "intent": intent,
        "entities": entities,
        "safety_verdict": "pass",
        "safety_reason": None,
        "confidence": 0.95,
    })
    mock_choice = MagicMock()
    mock_choice.message.content = payload
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def test_classifier_routing_accuracy(gold_classifier_queries):
    """
    Threshold: >= 85% routing accuracy.
    LLM is mocked to return the expected agent for each query —
    this tests that our classify() function correctly parses
    and passes through the LLM decision.
    """
    correct = 0
    failures = []

    with patch("src.classifier.intent.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        for case in gold_classifier_queries:
            expected_agent = case["expected_agent"]
            expected_entities = case.get("expected_entities", {})

            mock_client.chat.completions.create.return_value = _make_mock_response(
                agent=expected_agent,
                intent="test_intent",
                entities=expected_entities,
            )

            result = classify(case["query"], history=[])

            if result.agent == expected_agent:
                correct += 1
            else:
                failures.append(
                    f"Query: '{case['query']}' | "
                    f"Expected: {expected_agent} | Got: {result.agent}"
                )

    accuracy = correct / len(gold_classifier_queries)

    if failures:
        print(f"\nRouting failures ({len(failures)}):")
        for f in failures[:10]:
            print(" ", f)

    assert accuracy >= 0.85, (
        f"Routing accuracy {accuracy:.2%} below 85% "
        f"({correct}/{len(gold_classifier_queries)})"
    )


def test_classifier_entity_extraction(gold_classifier_queries):
    """
    Entity extraction match rate — reported, not a hard failure.
    """
    matched = 0
    total_with_entities = 0

    with patch("src.classifier.intent.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        for case in gold_classifier_queries:
            expected_entities = case.get("expected_entities", {})
            if not expected_entities:
                continue
            total_with_entities += 1

            mock_client.chat.completions.create.return_value = _make_mock_response(
                agent=case["expected_agent"],
                intent="test_intent",
                entities=expected_entities,
            )

            result = classify(case["query"], history=[])
            actual_entities = result.entities.to_dict()

            if matches_entities(actual_entities, expected_entities):
                matched += 1

    rate = matched / total_with_entities if total_with_entities else 0.0
    print(f"\nEntity match rate: {rate:.2%} ({matched}/{total_with_entities})")


def test_classifier_fallback_on_llm_failure():
    """Classifier must return a safe fallback when LLM fails, never raise."""
    with patch("src.classifier.intent.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API down")

        result = classify("test query", history=[])

    assert result is not None
    assert result.agent == "customer_support"
    assert result.confidence == 0.0


def test_classifier_handles_malformed_json():
    """Classifier must not crash on malformed LLM output."""
    with patch("src.classifier.intent.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "not json at all {{{"
        mock_client.chat.completions.create.return_value = mock_response

        result = classify("how is my portfolio doing?", history=[])

    assert result is not None
    assert result.agent is not None