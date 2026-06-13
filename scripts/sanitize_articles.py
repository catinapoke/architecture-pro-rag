#!/usr/bin/env python3
"""Replace Star Wars terms in knowledge_base/ using terms_map.json."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = ROOT / "knowledge_base"
MAP_PATH = ROOT / "terms_map.json"
MANIFEST_PATH = KNOWLEDGE_BASE_DIR / "manifest.json"

# High-risk markers used by --check when not present as map keys.
EXTRA_FORBIDDEN = [
    r"\bStar Wars\b",
    r"\bWookieepedia\b",
    r"\bLucasfilm\b",
    r"\bDisney\+?\b",
    r"\bJedi\b",
    r"\bSith\b",
    r"\bDarth\b",
    r"\bSkywalker\b",
    r"\bPalpatine\b",
    r"\bBBY\b",
    r"\bABY\b",
    r"\bThe Force\b",
    r"\bMay the Force be with you\b",
    r"\bDeath Star\b",
    r"\blightsaber\b",
    r"\bstormtrooper",
    r"\bMandalorian\b",
    r"\bCoruscant\b",
    r"\bTatooine\b",
    r"\bNaboo\b",
    r"\bHoth\b",
    r"\bMillennium Falcon\b",
    r"\bR2-D2\b",
    r"\bC-3PO\b",
    r"\bBB-8\b",
    r"\bClone Wars\b",
    r"\bGalactic Republic\b",
    r"\bGalactic Empire\b",
    r"\bRebel Alliance\b",
    r"\bFirst Order\b",
    r"\bKnights of the Old Republic\b",
    r"\bThe Mandalorian\b",
    r"\bYoda\b",
    r"\bVader\b",
    r"\bChewbacca\b",
    r"\bHan Solo\b",
    r"\bLeia\b",
    r"\bAnakin\b",
    r"\bLuke\b",
    r"\bWampa\b",
    r"\bWookiee\b",
    r"\bHutt\b",
    r"\bJabba\b",
    r"\bImperials\b",
    r"\bDroid\b",
    r"\bBB-8\b",
    r"\bBoba Fett\b",
    r"\bAhsoka\b",
    r"\bHyperspace\b",
    r"\bLEGO Star Wars\b",
    r"\bBonadan\b",
    r"\bOuter Rim\b",
    r"\bCore Worlds\b",
    r"\bKorriban\b",
    r"\bRevan\b",
    r"\bDooku\b",
    r"\bSeparatist\b",
    r"\bConfederacy\b",
    r"\bBith\b",
    r"\bRodian\b",
    r"\bTwi'lek\b",
    r"\bGrogu\b",
    r"\bKamino\b",
    r"\bMustafar\b",
    r"\bTaris\b",
    r"\bFerrix\b",
    r"\bJango Fett\b",
    r"\bBattle of Yavin\b",
    r"\bMos Espa\b",
    r"\bMos Eisley\b",
    r"\bEbon Hawk\b",
    r"\bCarth Onasi\b",
    r"\bMeetra Surik\b",
    r"\bEpisode I\b",
    r"\bEpisode IV\b",
    r"\bEpisode VI\b",
    r"\bEpisode VII\b",
    r"\bMos Espan\b",
    r"\bMos Eisden\b",
    r"\bPodrace\b",
    r"\bblaster\b",
    r"\bGalven\b",
    r"\bDamask\b",
    r"\bKrennic\b",
    r"\bAndor\b",
    r"\bDjarin\b",
    r"\bRey\b",
    r"\bNihil\b",
    r"\bHero of Tython\b",
    r"\bHK-50\b",
    r"\bT3-M4\b",
    r"\bSolo:\b",
    r"\bmicroblasters\b",
    r"\bLeviathan\b",
    r"\bBaran Do\b",
    r"\bPadawans\b",
    r"\bVergere\b",
    r"\bColossus\b",
    r"\bXiono\b",
    r"\bGerrera\b",
    r"\bErso\b",
    r"\bPanaka\b",
    r"\bAmidala\b",
]


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE)
    slug = re.sub(r"[\s_]+", "-", slug.strip())
    return slug[:120] or "article"


def load_map() -> dict[str, str]:
    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("terms_map.json must be a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def compile_patterns(term_map: dict[str, str]) -> list[tuple[re.Pattern[str], str, str]]:
    pairs: list[tuple[re.Pattern[str], str, str]] = []
    for source, target in term_map.items():
        if not source or source == target:
            continue
        escaped = re.escape(source)
        # Optional markdown italics and possessive suffixes.
        flags = 0 if source[:1].isupper() else re.IGNORECASE
        pattern = re.compile(
            rf"(?<![\w'’-])(?:_)?{escaped}(?:'s|'s|'|s')?(?:_)?(?![\w'’-])",
            flags,
        )
        pairs.append((pattern, source, target))
    pairs.sort(key=lambda item: len(item[1]), reverse=True)
    return pairs


def _cased_replacement(value: str, replacement: str) -> str:
    possessive = ""
    core = value
    for suffix in ("'s", "'s", "s'", "s'", "'"):
        if core.endswith(suffix) and suffix != "'":
            possessive = suffix
            core = core[: -len(suffix)]
            break
    if core.endswith("'") and not possessive:
        possessive = "'"
        core = core[:-1]

    if replacement.endswith("'s") or replacement.endswith("'s"):
        possessive = ""
    elif replacement.endswith("s'") or replacement.endswith("s'"):
        possessive = ""

    if replacement.islower():
        out = replacement
    elif core.isupper():
        out = replacement.upper()
    elif core[:1].isupper():
        parts = replacement.split(" ")
        out = " ".join(p[:1].upper() + p[1:] if p else p for p in parts)
    else:
        out = replacement

    if possessive == "'":
        out += "'"
    elif possessive in ("s'", "s'"):
        # Plural possessive: append to replacement if it doesn't already end in s.
        if not out.endswith("s"):
            out += "s"
        out += "'" if possessive == "s'" else "'"
    else:
        out += possessive
    return out


def cleanup_artifacts(text: str) -> str:
    fixes = [
        (r"'s's\b", "'s"),
        (r"s''\b", "s'"),
        (r"\bLord Lord\b", "Lord"),
        (r"\bpersuaded to Concord against\b", "persuaded to rebel against"),
        (r"\bConcord against\b", "rebel against"),
        (r"\ban Dominion\b", "a Dominion"),
        (r"\ba Umbral\b", "an Umbral"),
        (r"\bgreater Vanguard to pain\b", "greater resistance to pain"),
        (r"\bVanguard fighter\b", "insurgent fighter"),
        (r"\bstrike Flux of\b", "strike force of"),
        (r"\btry and Flux him\b", "try and compel him"),
        (r"\banti-Ghorman\b", "anti-Ghoran"),
        (r"\bDorlen'shan\b", "Dorval'shan"),
        (r"\bon Dorlen—\b", "on Dorval—"),
        (r"\bto Dorlen to\b", "to Dorval to"),
        (r"\bplanet Dorlen\b", "planet Dorval"),
        (r"\bDorlen's helium\b", "Dorval's helium"),
        (r"\bSolo was\b", "Solen was"),
        (r"\bSolo would\b", "Solen would"),
        (r"\bSolo eventually\b", "Solen eventually"),
        (r"\bSolo to\b", "Solen to"),
        (r"\bLaborotories\b", "Laboratories"),
        (r"_Barrier_ -class", "_Barrier_-class"),
        (r"\buvaks\b", "uvax"),
        (r"―17 and 35\b", "―Tordrasar 17 and Mirsol-Mirwen"),
        (r"(?<!Tordrasar )\b17 was a human male replica\b", "Tordrasar 17 was a human male replica"),
        (r"(?<!Tordrasar )\b17 was deployed\b", "Tordrasar 17 was deployed"),
        (r"\bOn Kryon, (?<!Tordrasar )17 and fellow trooper\b", "On Kryon, Tordrasar 17 and fellow trooper"),
        (r"\bYorpraor,\s*17,\s*35,", "Yorpraor, Tordrasar 17, Mirsol-Mirwen,"),
        (r"(?<!Tordrasar )\b17 stood\b", "Tordrasar 17 stood"),
        (r"(?<!Tordrasar )\b17 wore\b", "Tordrasar 17 wore"),
        (r"\bTordrasar Tordrasar 17\b", "Tordrasar 17"),
        (r"\bthe Bavos-I\b", "the Lyrka-Nistor"),
        (r"\bThe Bavos-I\b", "The Lyrka-Nistor"),
        (r"\bOne such Bavos-I\b", "One such Lyrka-Nistor"),
        (r"\bsuccessor to the Bavos-I\b", "successor to the Lyrka-Nistor"),
        (r"\bversion of the Bavos-I\b", "version of the Lyrka-Nistor"),
        (r"\b25,053BFC\b", "25,053 BFC"),
        (r"\n## trade chits\n\n\*\*By type\*\*\n\nCrew\s*$", ""),
    ]
    for pattern, repl in fixes:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def apply_replacements(text: str, patterns: list[tuple[re.Pattern[str], str, str]]) -> str:
    for pattern, _source, replacement in patterns:
        text = pattern.sub(
            lambda m, r=replacement: _cased_replacement(m.group(0), r),
            text,
        )
    return cleanup_artifacts(text)


def extract_title(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "article"


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(entries: list[dict]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def collect_article_paths() -> list[Path]:
    return sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))


def scan_forbidden(text: str, term_map: dict[str, str]) -> list[str]:
    findings: list[str] = []
    seen: set[str] = set()

    for source in sorted(term_map, key=len, reverse=True):
        flags = 0 if source[:1].isupper() else re.IGNORECASE
        if re.search(
            rf"(?<![\w'’-]){re.escape(source)}(?![\w'’-])",
            text,
            flags,
        ):
            if source.lower() not in seen:
                seen.add(source.lower())
                findings.append(source)

    for pattern in EXTRA_FORBIDDEN:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            token = match.group(0)
            if token.lower() not in seen:
                seen.add(token.lower())
                findings.append(token)
    return findings


def run_check(term_map: dict[str, str]) -> int:
    corpus_parts: list[str] = []
    for path in collect_article_paths():
        corpus_parts.append(path.read_text(encoding="utf-8"))
    if MANIFEST_PATH.exists():
        corpus_parts.append(MANIFEST_PATH.read_text(encoding="utf-8"))

    corpus = "\n".join(corpus_parts)
    findings = scan_forbidden(corpus, term_map)
    if findings:
        print("Forbidden Star Wars markers still present:")
        for item in sorted(findings, key=str.lower):
            print(f"  - {item}")
        return 1
    print("Check passed: no forbidden Star Wars markers found.")
    return 0


def sanitize(dry_run: bool = False) -> None:
    term_map = load_map()
    patterns = compile_patterns(term_map)
    manifest = load_manifest()
    manifest_by_index = {entry["index"]: entry for entry in manifest}

    planned: list[tuple[Path, Path, str, int | None]] = []

    for path in collect_article_paths():
        original = path.read_text(encoding="utf-8")
        sanitized = apply_replacements(original, patterns)
        title = extract_title(sanitized)

        index_match = re.match(r"^(\d+)-", path.name)
        index = int(index_match.group(1)) if index_match else None
        prefix = f"{index:02d}-" if index is not None else ""
        new_name = f"{prefix}{slugify(title)}.md"
        new_path = path.with_name(new_name)
        planned.append((path, new_path, title, index))

    if dry_run:
        print("Dry run: planned transformations")
        for old, new, title, _ in planned:
            if old.name != new.name:
                print(f"  rename: {old.name} -> {new.name}")
            else:
                print(f"  update: {old.name} ({title})")
        return

    temp_outputs: dict[Path, str] = {}
    for old_path, new_path, title, index in planned:
        content = apply_replacements(old_path.read_text(encoding="utf-8"), patterns)
        temp_outputs[new_path] = content

    for old_path in collect_article_paths():
        if old_path not in {new for _, new, _, _ in planned}:
            old_path.unlink(missing_ok=True)

    new_manifest: list[dict] = []
    for old_path, new_path, title, index in planned:
        new_path.write_text(temp_outputs[new_path], encoding="utf-8")
        if old_path != new_path and old_path.exists():
            old_path.unlink(missing_ok=True)
        if index is not None:
            new_manifest.append({"index": index, "title": title, "file": new_path.name})

    new_manifest.sort(key=lambda item: item["index"])
    save_manifest(new_manifest)
    print(f"Sanitized {len(new_manifest)} articles in {KNOWLEDGE_BASE_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanitize Star Wars terms in knowledge_base/")
    parser.add_argument("--check", action="store_true", help="Scan corpus for forbidden markers")
    parser.add_argument("--dry-run", action="store_true", help="Preview file renames without writing")
    args = parser.parse_args()

    if not MAP_PATH.exists():
        print(f"Missing map file: {MAP_PATH}", file=sys.stderr)
        return 2

    term_map = load_map()
    if args.check:
        return run_check(term_map)
    sanitize(dry_run=args.dry_run)
    return run_check(term_map)


if __name__ == "__main__":
    raise SystemExit(main())
