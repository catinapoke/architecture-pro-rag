#!/usr/bin/env python3
"""Unit tests for RAG evaluation helpers."""

from __future__ import annotations

import unittest

import test_rag_bot


class RagBotEvaluationTests(unittest.TestCase):
    def test_passing_judge_requires_correct_result(self) -> None:
        judge = {"correct": False, "fullness_score": 1.0}

        self.assertFalse(test_rag_bot.is_passing_judge(judge))

    def test_passing_judge_requires_minimum_score(self) -> None:
        judge = {"correct": True, "fullness_score": 0.74}

        self.assertFalse(test_rag_bot.is_passing_judge(judge))

    def test_passing_judge_accepts_correct_result_at_minimum_score(self) -> None:
        judge = {"correct": True, "fullness_score": 0.75}

        self.assertTrue(test_rag_bot.is_passing_judge(judge))

    def test_passing_judge_accepts_correct_missing_information_refusal(self) -> None:
        judge = {"correct": True, "fullness_score": 0.0}

        self.assertTrue(test_rag_bot.is_passing_judge(judge, expect_answer=False))


if __name__ == "__main__":
    unittest.main()
