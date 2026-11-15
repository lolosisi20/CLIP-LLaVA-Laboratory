import re
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, Field

from ..config import settings
from ..llm import ask_json, parallel_map
from ..paper import Paper
from ..text import contains_quote, normalized
from ..trace import log_step

CHUNK_CHARACTERS = 3000
MIN_KEPT = 0.6  # share of the quoted characters a fix must keep


class LanguageIssue(BaseModel):
    quote: str = Field(description="the faulty words, copied exactly, at most one sentence")
    problem: str
    suggestion: str = Field(description="the corrected words")
    category: Literal["grammar", "spelling", "vocabulary", "punctuation"]


class LanguageReview(BaseModel):
    issues: list[LanguageIssue]


LANGUAGE_SYSTEM = """You are a copy editor for scientific English. Report only objective errors: grammar (agreement,
tense, articles, prepositions), spelling, punctuation, and vocabulary (a word used with the wrong meaning).
Do NOT rewrite correct sentences for style, concision or preference: if a sentence is grammatical and its words are
used correctly, it is not an issue. The text was extracted from a PDF: ignore broken math, symbols, citation numbers,
table cells, hyphenation and line-break artifacts. Most paragraphs of a published paper have no errors; return an
empty list for them."""


def review_chunk(pair: tuple[str, str]) -> list[dict]:
    section, chunk = pair
    review = ask_json(f"Section: {section}\n---\n{chunk}", LanguageReview, agent="language", system=LANGUAGE_SYSTEM, max_tokens=8000, think=True)
    return [{**issue.model_dump(), "section": section} for issue in review.issues] if review else []


def is_small_edit(quote: str, fix: str) -> bool:
    """Most of the quote survives in the fix, and the fix is not a longer rewrite."""
    quote, fix = normalized(quote), normalized(fix).strip("'\"‘’“” .")
    kept = sum(block.size for block in SequenceMatcher(None, quote, fix).get_matching_blocks()) / max(1, len(quote))
    return quote.strip(" .") != fix and kept >= MIN_KEPT and len(fix) <= 2 * len(quote) + 30


def is_real_issue(issue: dict, paper_text: str) -> bool:
    """Quoted from the paper, in prose rather than garbled math, and fixed by a small edit, not a rewrite."""
    is_quoted = contains_quote(paper_text, issue["quote"])
    is_prose = not re.search(r"[^\x00-\x7F’“”–—×]", issue["quote"])
    return is_quoted and is_prose and is_small_edit(issue["quote"], issue["suggestion"])


def review_language(paper: Paper) -> dict:
    chunks = paper.chunks(CHUNK_CHARACTERS)
    found = [issue for issues in parallel_map(review_chunk, chunks, settings.protocol["review"]["parallel_agents"]) for issue in issues]
    real = [issue for issue in found if is_real_issue(issue, paper.text)]
    unique = list({normalized(issue["quote"]): issue for issue in real}.values())
    log_step("language", f"{len(found)} issues reported, {len(unique)} kept (quoted from the paper, with a real change)")
    return {"issues": unique, "reported": len(found)}
