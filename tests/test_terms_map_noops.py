#!/usr/bin/env python3
"""Regression tests for replacement dictionaries."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_literal_dict(path: Path, name: str) -> dict[str, str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


class TermsMapNoopTests(unittest.TestCase):
    def test_source_dictionaries_do_not_map_terms_to_themselves(self) -> None:
        dictionaries = {
            "MAP": load_literal_dict(ROOT / "scripts" / "generate_terms_map.py", "MAP"),
            "EXPANSION": load_literal_dict(ROOT / "scripts" / "terms_expansion.py", "EXPANSION"),
            "PASS2": load_literal_dict(ROOT / "scripts" / "terms_pass2.py", "PASS2"),
        }

        noops = {
            f"{dict_name}.{source}": target
            for dict_name, mapping in dictionaries.items()
            for source, target in mapping.items()
            if source == target
        }

        self.assertEqual({}, noops)

    def test_knowledge_base_does_not_keep_partially_replaced_entity_names(self) -> None:
        corpus = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((ROOT / "knowledge_base").glob("*.md"))
        )

        forbidden_fragments = [
            "Replica Trooper **17**",
            "_Altor_ -class",
            "Bavos-I",
            "Bavos-II",
            "HFY-3920",
            "A95 Stingbeams",
            "A95 Stingbeam",
            "T3-series Utility Automaton",
            "Yorpraor, 17, 35",
            "Tordrasar Tordrasar 17",
            "Stingbeams",
            "Stingbeam's",
            "3C automaton",
            "Ablund sold",
            "Ablund aided",
            "Chains Worlds Theorem",
            "Saal's apparent",
            "Saal alive",
            "McCaig's",
            "Chapman provided",
            "Chapman called",
            "Chapman stated",
            "Veil Menace soundtrack",
            "Asner later",
            "Asner's",
            "Carano was",
            "Carano filed",
            "Tagrin would",
            "Tagrin's character",
            "Hanar's Archive Seal",
            "25,053BFC",
            "Auman felt",
            "Auman further",
            "Doza would",
            "Burke later",
            "Burke described",
            "Hignight described",
            "Hignight lamented",
            "Wyatt revealed",
            "In A Future, Far Beyond The Veil …",
            "In A Future, Far Beyond The Veil",
            "Yorpraor",
            "For other uses, see **Ed**.",
            "## trade chits\n\n**By type**\n\nCrew",
        ]

        found = [fragment for fragment in forbidden_fragments if fragment in corpus]
        self.assertEqual([], found)


if __name__ == "__main__":
    unittest.main()
