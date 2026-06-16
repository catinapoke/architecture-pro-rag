#!/usr/bin/env python3
"""Unit tests for index build helpers."""

from __future__ import annotations

import unittest
import sys

try:
    import build_index
    IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:
    build_index = None  # type: ignore[assignment]
    IMPORT_ERROR = exc


@unittest.skipIf(build_index is None, f"Required dependency is missing: {IMPORT_ERROR}")
class BuildIndexHelpersTests(unittest.TestCase):
    def test_build_index_does_not_import_faiss_until_needed(self) -> None:
        self.assertNotIn("faiss", sys.modules)

    def test_load_models_warms_embeddings_before_creating_index(self) -> None:
        events: list[str] = []

        class FakeEmbeddings:
            def __init__(self, **_kwargs: object) -> None:
                events.append("init")

            def embed_query(self, _text: str) -> list[float]:
                events.append("warmup")
                return [0.0]

        original_embeddings = build_index.HuggingFaceEmbeddings
        original_create_empty_index = build_index.create_empty_index
        try:
            build_index.HuggingFaceEmbeddings = FakeEmbeddings
            build_index.create_empty_index = lambda: events.append("index") or object()

            build_index.load_models()
        finally:
            build_index.HuggingFaceEmbeddings = original_embeddings
            build_index.create_empty_index = original_create_empty_index
            build_index.embeddings = None
            build_index.index = None

        self.assertEqual(["init", "warmup", "index"], events)

    def test_make_chunk_id_is_stable_for_same_input(self) -> None:
        first = build_index.make_chunk_id("article.md", 3)
        second = build_index.make_chunk_id("article.md", 3)
        third = build_index.make_chunk_id("article.md", 4)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertGreaterEqual(first, 0)

    def test_build_file_metadata_uses_chunk_objects(self) -> None:
        processed = build_index.ProcessedFile(
            file_name="article.md",
            file_sha256="file-sha",
            pieces=["chunk-1", "chunk-2"],
            vectors=[[1.0], [2.0]],
            chunk_ids=[11, 12],
            chunk_shas=["sha-1", "sha-2"],
        )

        metadata = build_index.build_file_metadata(processed)

        self.assertEqual("file-sha", metadata["sha256"])
        self.assertEqual(
            [
                {"id": 11, "sha": "sha-1", "index": 0},
                {"id": 12, "sha": "sha-2", "index": 1},
            ],
            metadata["chunks"],
        )


if __name__ == "__main__":
    unittest.main()
