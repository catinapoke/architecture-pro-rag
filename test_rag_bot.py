#!/usr/bin/env python3
"""Golden-set evaluation for the RAG bot.

The script asks questions from golden_questions.json, saves every answer to a
JSONL log, and uses the configured OpenAI-compatible LLM as a semantic judge.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
from pydantic import BaseModel, Field, ValidationError


DEFAULT_GOLDEN_PATH = Path("golden_questions.json")
DEFAULT_LOG_PATH = Path("logs") / "rag_eval.jsonl"
MIN_PASSING_SCORE = 0.75


@dataclass(frozen=True)
class GoldenQuestion:
    id: int
    section: str
    question: str
    expected: str
    expect_answer: bool


def load_golden_questions(path: Path) -> list[GoldenQuestion]:
    raw_questions = json.loads(path.read_text(encoding="utf-8"))["questions"]
    return [
        GoldenQuestion(
            id=int(raw_item.get("id", index)),
            section=str(raw_item["section"]),
            question=str(raw_item["question"]),
            expected=str(raw_item["expected"]),
            expect_answer=raw_item["expect_answer"],
        )
        for index, raw_item in enumerate(raw_questions, start=1)
    ]


class JudgeResult(BaseModel):
    answer_found: bool
    correct: bool
    fullness_score: float = Field(ge=0.0, le=1.0)
    reason: str


def _normalize_fullness_score(raw_score: Any) -> float:
    score = float(raw_score)
    # Some local models respond in a 0..10 scale despite prompt/schema hints.
    if score > 1.0 and score <= 10.0:
        score = score / 10.0
    return max(0.0, min(1.0, score))


def _extract_assistant_payload(message: Any) -> str:
    content = getattr(message, "content", "") or ""
    if content:
        return content
    return getattr(message, "reasoning_content", "") or ""


def _coerce_judge_result(parsed: Any, payload: str) -> tuple[JudgeResult | None, str]:
    if isinstance(parsed, JudgeResult):
        normalized = parsed.model_copy(update={"fullness_score": _normalize_fullness_score(parsed.fullness_score)})
        return normalized, ""
    if not payload:
        return None, ""
    try:
        data = json.loads(payload)
        if isinstance(data, dict) and "fullness_score" in data:
            data["fullness_score"] = _normalize_fullness_score(data["fullness_score"])
        return JudgeResult.model_validate(data), ""
    except (TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
        return None, str(exc)


def is_passing_judge(judge: dict[str, Any], expect_answer: bool = True) -> bool:
    if not bool(judge["correct"]):
        return False
    if not expect_answer:
        return True
    return float(judge["fullness_score"]) >= MIN_PASSING_SCORE


def judge_answer(client: Any, model: str, item: GoldenQuestion, answer: str) -> dict[str, Any]:
    expected_behavior = (
        "The bot must answer the question using the expected facts."
        if item.expect_answer
        else "The bot must refuse because the knowledge base does not contain enough information."
    )
    judge_prompt = f"""
You are evaluating a RAG bot answer.
Compare the bot answer with the expected answer semantically, not word-for-word.

Rules:
- Return only a valid JSON object.
- Use this schema:
  {{
    "answer_found": boolean,
    "correct": boolean,
    "fullness_score": number,
    "reason": "short explanation in Russian"
  }}
- fullness_score must be from 0.0 to 1.0.
- For missing-information questions, correct=true only when the bot clearly says that it does not know
  or that the knowledge base has no information. If it invents the expected fact, correct=false.

Expected behavior: {expected_behavior}
Question: {item.question}
Expected answer: {item.expected}
Bot answer: {answer}
""".strip()

    try:
        response = client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "developer", "content": judge_prompt},
                {"role": "user", "content": "Evaluate the bot answer and return JSON."},
            ],
            temperature=0,
            response_format=JudgeResult,
        )
    except Exception as exc:
        return {
            "answer_found": False,
            "correct": False,
            "fullness_score": 0.0,
            "reason": f"LLM judge structured parsing failed: {exc}",
            "raw_judge_answer": "",
        }

    message = response.choices[0].message
    content = _extract_assistant_payload(message)
    parsed, parse_error = _coerce_judge_result(getattr(message, "parsed", None), content)
    if parsed is None:
        refusal = getattr(message, "refusal", "")
        refusal_reason = f" Refusal: {refusal}" if refusal else ""
        parse_error_reason = f" Parse error: {parse_error}" if parse_error else ""
        return {
            "answer_found": False,
            "correct": False,
            "fullness_score": 0.0,
            "reason": f"LLM judge did not return structured output.{refusal_reason}{parse_error_reason}",
            "raw_judge_answer": content,
        }

    return {
        "answer_found": bool(parsed.answer_found),
        "correct": bool(parsed.correct),
        "fullness_score": float(parsed.fullness_score),
        "reason": str(parsed.reason),
        "raw_judge_answer": content,
    }


def build_record(
    item: GoldenQuestion,
    answer: str,
    judge: dict[str, Any],
    chunks: list[Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    sources = [
        {
            "file": chunk.file,
            "chunk_index": chunk.index,
            "distance": chunk.distance,
        }
        for chunk in chunks
    ]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question_id": item.id,
        "section": item.section,
        "question": item.question,
        "expected": item.expected,
        "expect_answer": item.expect_answer,
        "answer": answer,
        "answer_length": len(answer),
        "found_chunks": len(chunks) > 0,
        "sources": sources,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "judge": judge,
        "status": "pass" if is_passing_judge(judge, expect_answer=item.expect_answer) else "fail",
    }


def evaluate(args: argparse.Namespace) -> int:
    from stdio_bot import client, get_response, model
    from query_index import find_related_chunks

    questions = load_golden_questions(args.golden_path)
    if args.limit is not None:
        questions = questions[: args.limit]

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    with args.log_path.open("w", encoding="utf-8") as log_file:
        for item in questions:
            start = time.time()
            chunks = find_related_chunks(item.question, top_k=args.top_k)
            answer = get_response(item.question)
            judge = judge_answer(client, model, item, answer)
            record = build_record(item, answer, judge, chunks, time.time() - start)
            records.append(record)
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()

            score = judge["fullness_score"]
            status = record["status"].upper()
            print(f"{item.id:02d}. {status} score={score:.2f} section={item.section}: {judge['reason']}")

    passed = sum(1 for record in records if record["status"] == "pass")
    total = len(records)
    accuracy = passed / total if total else 0.0
    print(f"\nPassed {passed}/{total}; accuracy={accuracy:.2%}; log={args.log_path}")

    if not args.no_fail and accuracy < args.fail_under:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-path", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fail-under", type=float, default=0.8)
    parser.add_argument("--no-fail", action="store_true", help="Always exit with code 0 after evaluation.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(evaluate(parse_args()))
