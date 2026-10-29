#!/usr/bin/env python3
"""Fail when a static-site link or runtime asset points to a missing local file."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1] / "docs"
TEXT_SUFFIXES = {".html", ".js", ".css", ".md"}
ATTR = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']")


def local_target(source: Path, raw: str) -> Path | None:
    value = raw.strip()
    if not value or value.startswith(("#", "data:", "mailto:", "javascript:")):
        return None
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        return None
    path = unquote(parsed.path)
    if not path:
        return None
    if path.startswith("/"):
        prefix = "/LazyAttention/"
        path = path[len(prefix) :] if path.startswith(prefix) else path.lstrip("/")
        return ROOT / path
    return source.parent / path


def main() -> int:
    missing: list[tuple[Path, str, Path]] = []
    checked = 0
    for source in ROOT.rglob("*"):
        if not source.is_file() or source.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = source.read_text(encoding="utf-8")
        for raw in ATTR.findall(text):
            target = local_target(source, raw)
            if target is None:
                continue
            checked += 1
            if not target.exists():
                missing.append((source.relative_to(ROOT), raw, target))
    if missing:
        for source, raw, target in missing:
            print(f"missing: {source}: {raw} -> {target}", file=sys.stderr)
        return 1
    print(f"checked {checked} local references under {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
