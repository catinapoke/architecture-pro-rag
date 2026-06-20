#!/usr/bin/env python3
"""Unit tests for update-index diff helpers."""

from __future__ import annotations

import unittest

try:
    import update_index
    IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:
    update_index = None  # type: ignore[assignment]
    IMPORT_ERROR = exc


@unittest.skipIf(update_index is None, f"Required dependency is missing: {IMPORT_ERROR}")
class UpdateIndexHelpersTests(unittest.TestCase):
    def test_find_file_changes_detects_new_changed_removed(self) -> None:
        stored = {
            "old.md": {"sha256": "old-sha", "chunks": []},
            "same.md": {"sha256": "same-sha", "chunks": []},
            "changed.md": {"sha256": "before", "chunks": []},
        }
        current_sha = {
            "same.md": "same-sha",
            "changed.md": "after",
            "new.md": "new-sha",
        }

        new_files, changed_files, removed_files = update_index.find_file_changes(stored, current_sha)

        self.assertEqual(["new.md"], new_files)
        self.assertEqual(["changed.md"], changed_files)
        self.assertEqual(["old.md"], removed_files)

    def test_get_file_chunk_ids_returns_int_list(self) -> None:
        file_entry = {
            "sha256": "file-sha",
            "chunks": [
                {"id": 11, "sha": "x", "index": 0},
                {"id": "12", "sha": "y", "index": 1},
            ],
        }

        self.assertEqual([11, 12], update_index.get_file_chunk_ids(file_entry))


if __name__ == "__main__":
    unittest.main()
