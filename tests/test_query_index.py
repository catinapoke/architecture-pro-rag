#!/usr/bin/env python3
"""Unit tests for query metadata lookup helpers."""

from __future__ import annotations

import unittest

try:
    import query_index
    IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:
    query_index = None  # type: ignore[assignment]
    IMPORT_ERROR = exc


@unittest.skipIf(query_index is None, f"Required dependency is missing: {IMPORT_ERROR}")
class QueryIndexHelpersTests(unittest.TestCase):
    def test_build_chunk_lookup_flattens_file_chunks(self) -> None:
        chunks_data = {
            "schema_version": 2,
            "files": {
                "a.md": {
                    "sha256": "file-a",
                    "chunks": [
                        {"id": 101, "sha": "a1", "index": 0},
                    ],
                },
                "b.md": {
                    "sha256": "file-b",
                    "chunks": [
                        {"id": 202, "sha": "b1", "index": 3},
                    ],
                },
            },
        }

        lookup = query_index.build_chunk_lookup(chunks_data)

        self.assertEqual({"file": "a.md", "index": 0, "sha": "a1"}, lookup[101])
        self.assertEqual({"file": "b.md", "index": 3, "sha": "b1"}, lookup[202])


if __name__ == "__main__":
    unittest.main()
