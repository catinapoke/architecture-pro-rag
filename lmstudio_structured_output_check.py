#!/usr/bin/env python3
"""Smoke-test LM Studio structured output behavior.

This script sends direct OpenAI-compatible requests to LM Studio and prints:
1) raw response content,
2) parsed JSON payload,
3) Pydantic validation result.

It helps confirm whether structured output works for your currently loaded model.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


class JudgeResult(BaseModel):
    answer_found: bool
    correct: bool
    fullness_score: float = Field(ge=0.0, le=1.0)
    reason: str


def normalize_fullness_score(raw_score: Any) -> float:
    score = float(raw_score)
    if score > 1.0 and score <= 10.0:
        score = score / 10.0
    return max(0.0, min(1.0, score))


JSON_SCHEMA_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_result",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer_found": {"type": "boolean"},
                "correct": {"type": "boolean"},
                "fullness_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reason": {"type": "string"},
            },
            "required": ["answer_found", "correct", "fullness_score", "reason"],
        },
    },
}


def build_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "developer",
            "content": (
                "You are a strict evaluation assistant. "
                "Return only the requested structured result."
            ),
        },
        {
            "role": "user",
            "content": (
                "Question: Что такое veil ceremony?\n"
                "Expected answer: Ритуал имитации смерти для тайного вступления.\n"
                "Bot answer: Veil ceremony — это обряд имитации смерти для вступления в тайный орден.\n"
                "Evaluate and return structured fields."
            ),
        },
    ]


def request_json_schema(client: OpenAI, model: str) -> None:
    print("=== Request #1: chat.completions.create + response_format=json_schema ===")
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(),
        temperature=0,
        response_format=JSON_SCHEMA_RESPONSE_FORMAT,
    )

    message = response.choices[0].message
    content = message.content or ""
    reasoning_content = getattr(message, "reasoning_content", "") or ""
    payload_text = content or reasoning_content
    print("Raw content:", content)
    print("Reasoning content:", reasoning_content)
    print("Payload used for parsing:", payload_text)

    try:
        payload = json.loads(payload_text)
        print("JSON parsed:", payload)
    except json.JSONDecodeError as exc:
        print("JSON parse error:", exc)
        return

    try:
        validated = JudgeResult.model_validate(payload)
        print("Pydantic validated:", validated.model_dump())
    except ValidationError as exc:
        print("Pydantic validation error:", exc)
        if isinstance(payload, dict) and "fullness_score" in payload:
            payload["fullness_score"] = normalize_fullness_score(payload["fullness_score"])
            validated = JudgeResult.model_validate(payload)
            print("Pydantic validated after normalization:", validated.model_dump())


def request_sdk_parse(client: OpenAI, model: str) -> None:
    print("\n=== Request #2: chat.completions.parse + response_format=JudgeResult ===")
    response = client.chat.completions.parse(
        model=model,
        messages=build_messages(),
        temperature=0,
        response_format=JudgeResult,
    )

    message = response.choices[0].message
    print("Raw content:", message.content or "")
    print("Reasoning content:", getattr(message, "reasoning_content", "") or "")
    print("Refusal:", getattr(message, "refusal", None))
    print("Parsed object:", message.parsed)

    if message.parsed is not None:
        print("Parsed dump:", message.parsed.model_dump())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--api-key", default="lm-studio")
    parser.add_argument("--model", default="qwen/qwen-3.6-35b-a3b")
    parser.add_argument(
        "--mode",
        choices=("json-schema", "sdk-parse", "both"),
        default="both",
        help="Which check to run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    if args.mode in ("json-schema", "both"):
        request_json_schema(client, args.model)
    if args.mode in ("sdk-parse", "both"):
        request_sdk_parse(client, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
