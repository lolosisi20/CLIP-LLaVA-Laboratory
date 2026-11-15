import re
import threading
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from .config import settings
from .http import cached_get
from .llm import ask_json
from .retrieval import HybridIndex, Passage
from .text import clean, contains_quote, split_into_chunks

DIGEST_CHARACTERS = 12_000


@dataclass
class Source:
    id: str
    kind: str
    title: str
    url: str
    text: str
    summary: str = ""
    quotes: list[str] = field(default_factory=list)

    def excerpt(self) -> str:
        """What a judge sees of the source: the digest when there is one, else the start of the text itself."""
        if self.quotes or self.summary:
            quotes = "\n".join(f'  - "{quote}"' for quote in self.quotes) or "  (none)"
            return f"Summary by the reader (not quotable): {self.summary}\nQuotes from the source (cite these):\n{quotes}"
        return f"Text of the source (quotable):\n{self.text[:1500]}"

    def public(self) -> dict:
        return {"id": self.id, "kind": self.kind, "title": self.title, "url": self.url,
                "summary": self.summary, "quotes": self.quotes}


class Ledger:
    """Every source one validator read. Evidence may only quote text that is really in one of them."""

    def __init__(self):
        self.sources: list[Source] = []
        self.lock = threading.Lock()

    def add(self, kind: str, title: str, url: str, text: str, summary: str = "", quotes=()) -> Source:
        with self.lock:
            prefix = {"web": "S", "paper": "P", "literature": "L", "python": "C"}[kind]
            number = sum(source.kind == kind for source in self.sources) + 1
            source = Source(f"{prefix}{number}", kind, title, url, text, summary, list(quotes))
            self.sources.append(source)
            return source

    def find(self, source_id: str) -> Source | None:
        return next((source for source in self.sources if source.id == source_id.strip("[] ")), None)

    def locate(self, source_id: str, quote: str) -> Source | None:
        """The source that really contains the quote: the cited one, else any other source this validator read."""
        cited = self.find(source_id)
        candidates = ([cited] if cited else []) + self.sources
        return next((source for source in candidates if contains_quote(source.text, quote)), None)

    def listing(self, limit: int = 20_000) -> str:
        text = "\n\n".join(f"[{source.id}] {source.title} {source.url}\n{source.excerpt()}" for source in self.sources)
        return text[:limit]


# ── Search ──

def web_search(query: str, limit: int = 6) -> list[dict]:
    response = httpx.get(settings.search_url, params={"q": query, "max": limit, "category": "general"}, timeout=60)
    response.raise_for_status()
    return [{"title": hit.get("title", ""), "url": hit.get("url", ""), "snippet": hit.get("snippet", "")}
            for hit in response.json().get("sources", [])[:limit]]


# ── Reading a page into a digest ──

def page_text(url: str) -> tuple[str, str]:
    """(title, readable text) of a web page or PDF."""
    if re.match(r"https?://arxiv\.org/abs/", url):
        url = url.replace("/abs/", "/pdf/")
    body = cached_get(url)
    if body.lstrip().startswith("<"):
        soup = BeautifulSoup(body, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else url
        return title, clean(soup.get_text("\n"))
    return url, clean(body)


def passages_about(text: str, question: str) -> str:
    """The parts of a long source closest to the question, in reading order, up to DIGEST_CHARACTERS."""
    if len(text) <= DIGEST_CHARACTERS:
        return text
    chunks = split_into_chunks(text, 1200)
    index = HybridIndex([Passage(chunk, str(position)) for position, chunk in enumerate(chunks)])
    best = index.search(question, k=DIGEST_CHARACTERS // 1200)
    return "\n...\n".join(passage.text for passage in sorted(best, key=lambda passage: int(passage.source)))


class Digest(BaseModel):
    summary: str = Field(description="what the source says about the question, in 2-4 sentences")
    quotes: list[str] = Field(description="up to 5 sentences copied word for word from the source")


def digest_page(title: str, text: str, url: str, question: str, agent: str) -> tuple[Digest, int]:
    """The digest of a page for a question, and how many of its quotes were dropped for not being in the page."""
    excerpt = passages_about(text, question)
    prompt = f"""Question: {question}

Source: {title} ({url})
---
{excerpt}
---
Say what this source says about the question. Copy up to 5 sentences that bear on it exactly as written.
If it says nothing about the question, say so and give no quotes."""
    digest = ask_json(prompt, Digest, agent=agent, max_tokens=2500) or Digest(summary="(could not digest)", quotes=[])
    kept = [quote for quote in digest.quotes if contains_quote(text, quote)]
    dropped = len(digest.quotes) - len(kept)
    digest.quotes = kept
    return digest, dropped
