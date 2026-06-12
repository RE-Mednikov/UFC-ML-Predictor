from __future__ import annotations

from urllib.parse import urlparse


def stable_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    slug = parsed.path.rstrip("/").split("/")[-1]
    return slug or None


def event_id_from_url(url: str | None) -> str | None:
    return stable_id_from_url(url)


def fight_id_from_url(url: str | None) -> str | None:
    return stable_id_from_url(url)


def fighter_id_from_url(url: str | None) -> str | None:
    return stable_id_from_url(url)