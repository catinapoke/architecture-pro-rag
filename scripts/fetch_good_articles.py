#!/usr/bin/env python3
"""Fetch Good Articles from Wookieepedia and save as clean plain-text markdown."""

import html2text
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://starwars.fandom.com/api.php"
USER_AGENT = "Mozilla/5.0 (compatible; WookieepediaScraper/1.0; educational)"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
TARGET_COUNT = 40

SKIP_PREFIXES = (
    "Wookieepedia:",
    "Category:",
    "File:",
    "Template:",
    "Help:",
    "Special:",
    "User:",
    "Talk:",
    "Forum:",
)

SKIP_SECTIONS = {
    "contents",
    "behind the scenes",
    "appearances",
    "sources",
    "notes and references",
    "external links",
    "in other languages",
    "bibliography",
    "related categories",
}


def api_request(params: dict) -> dict:
    params["format"] = "json"
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE)
    slug = re.sub(r"[\s_]+", "-", slug.strip())
    return slug[:120] or "article"


def strip_html_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html).strip()


def clean_html(html: str) -> str:
    # Meta blocks: era icons, hatnotes, TOC, infoboxes, navboxes, references, media.
    patterns = [
        r"<div[^>]*id=\"title-eraicons\"[^>]*>.*?</div>",
        r"<div[^>]*class=\"[^\"]*hatnote[^\"]*\"[^>]*>.*?</div>",
        r"<div[^>]*class=\"[^\"]*dablink[^\"]*\"[^>]*>.*?</div>",
        r"<div[^>]*id=\"toc\"[^>]*>.*?</div>",
        r"<table[^>]*>.*?</table>",
        r"<aside[^>]*>.*?</aside>",
        r"<figure[^>]*>.*?</figure>",
        r"<div[^>]*class=\"[^\"]*navbox[^\"]*\"[^>]*>.*?</div>",
        r"<div[^>]*class=\"[^\"]*references[^\"]*\"[^>]*>.*?</div>",
        r"<div[^>]*class=\"[^\"]*catlinks[^\"]*\"[^>]*>.*?</div>",
        r"<ol[^>]*class=\"[^\"]*references[^\"]*\"[^>]*>.*?</ol>",
        r"<sup[^>]*>.*?</sup>",
        r"<span[^>]*class=\"[^\"]*mw-default-size[^\"]*\"[^>]*>.*?</span>",
        r"<span[^>]*class=\"[^\"]*mw-editsection[^\"]*\"[^>]*>.*?</span>",
    ]
    for pattern in patterns:
        html = re.sub(pattern, "", html, flags=re.DOTALL | re.IGNORECASE)

    # Drop non-content h2 sections.
    parts = re.split(r"(<h2[^>]*>.*?</h2>)", html, flags=re.DOTALL | re.IGNORECASE)
    if len(parts) > 1:
        kept = [parts[0]]
        for i in range(1, len(parts), 2):
            heading_html = parts[i]
            body = parts[i + 1] if i + 1 < len(parts) else ""
            heading = strip_html_tags(heading_html).lower()
            if heading in SKIP_SECTIONS:
                continue
            kept.append(heading_html)
            kept.append(body)
        html = "".join(kept)

    return html


def html_to_markdown(html: str) -> str:
    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.ignore_emphasis = False
    converter.body_width = 0
    converter.unicode_snob = True
    return converter.handle(html)


def clean_markdown(text: str) -> str:
    # Residual markdown links, citations, audio links, empty badges.
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[\]\([^)]*\)", "", text)
    text = re.sub(r"\[\[Source\]\]", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"▶️\s*\([^)]*\)", "", text)
    text = re.sub(r"—\s*Link\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"You may be looking for.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"This article is about.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^In other languages\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    # Table of contents rendered as numbered bullet list.
    text = re.sub(r"(?:^\s*\*\s+\d+\s+.+$\n?)+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_article_markdown(title: str) -> tuple[str, str]:
    data = api_request({
        "action": "parse",
        "page": title,
        "prop": "text|displaytitle",
        "disableeditsection": "true",
    })
    if "error" in data:
        raise ValueError(data["error"].get("info", str(data["error"])))

    parsed = data["parse"]
    display = strip_html_tags(parsed.get("displaytitle", title))
    html = clean_html(parsed["text"]["*"])
    body = clean_markdown(html_to_markdown(html))
    md = f"# {display}\n\n{body}\n"
    return display, md


def parse_good_articles_page() -> list[str]:
    data = api_request({
        "action": "parse",
        "page": "Wookieepedia:Good_articles",
        "prop": "text",
    })
    html = data["parse"]["text"]["*"]

    sections = re.split(r"<h2[^>]*>", html)
    section_articles: list[list[str]] = []

    for chunk in sections[1:]:
        links = re.findall(r'href="/wiki/([^"#?]+)"', chunk)
        arts: list[str] = []
        seen: set[str] = set()
        for link in links:
            t = urllib.parse.unquote(link.replace("_", " "))
            if any(t.startswith(p) for p in SKIP_PREFIXES):
                continue
            if t in seen:
                continue
            seen.add(t)
            arts.append(t)
        if arts:
            section_articles.append(arts)

    selected: list[str] = []
    indices = [0] * len(section_articles)
    while len(selected) < TARGET_COUNT:
        added = False
        for i, arts in enumerate(section_articles):
            if indices[i] < len(arts):
                title = arts[indices[i]]
                indices[i] += 1
                if title not in selected:
                    selected.append(title)
                    added = True
                if len(selected) >= TARGET_COUNT:
                    break
        if not added:
            break

    return selected[:TARGET_COUNT]


def load_manifest_titles() -> list[str]:
    manifest_path = OUTPUT_DIR / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [item["title"] for item in sorted(manifest, key=lambda x: x["index"])]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    titles = load_manifest_titles()
    if not titles:
        print("Parsing Good Articles page...")
        titles = parse_good_articles_page()
    else:
        print(f"Reprocessing {len(titles)} articles from manifest...")

    manifest = []
    for i, title in enumerate(titles, 1):
        print(f"[{i}/{len(titles)}] {title}")
        try:
            display, md = fetch_article_markdown(title)
            filename = f"{i:02d}-{slugify(display)}.md"
            (OUTPUT_DIR / filename).write_text(md, encoding="utf-8")
            manifest.append({"index": i, "title": display, "file": filename})
            time.sleep(0.25)
        except Exception as exc:
            print(f"  ! Error: {exc}")

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nDone: {len(manifest)} articles saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
