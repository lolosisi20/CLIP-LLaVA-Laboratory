import hashlib
import json
import time

import httpx

from .config import settings

HEADERS = {"User-Agent": "Mozilla/5.0 (paper-review research pipeline)"}


def cached_get(url: str, params: dict | None = None, as_json: bool = False, retries: int = 2, headers: dict | None = None):
    """GET with an on-disk cache, so a rerun reads the same pages and search results as the first run."""
    key = hashlib.sha1(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
    path = settings.cache_dir / "http" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["body"]

    response = None
    for attempt in range(retries + 1):
        response = httpx.get(url, params=params, headers=HEADERS | (headers or {}), timeout=30, follow_redirects=True)
        if response.status_code != 429:
            break
        time.sleep(3 * (attempt + 1))
    response.raise_for_status()

    body = response.json() if as_json else read_body(response)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"url": str(response.url), "body": body}, ensure_ascii=False), encoding="utf-8")
    return body


def read_body(response: httpx.Response) -> str:
    is_pdf = "pdf" in response.headers.get("content-type", "") or response.content[:4] == b"%PDF"
    if not is_pdf:
        return response.text
    import pymupdf
    document = pymupdf.open(stream=response.content, filetype="pdf")
    return "\n".join(page.get_text() for page in document)
