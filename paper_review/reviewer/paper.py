import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from .text import clean, split_into_chunks

END_OF_BODY = re.compile(r"\n\s*(References|Bibliography|Acknowledg(e)?ments)\s*\n", re.IGNORECASE)
APPENDIX_HEADING = re.compile(r"^[A-Z]\s")


@dataclass
class Section:
    title: str
    text: str


@dataclass
class Paper:
    title: str
    source: str
    sections: list[Section] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(f"## {section.title}\n{section.text}" for section in self.sections)

    @property
    def abstract(self) -> str:
        first = self.sections[0] if self.sections else Section("", "")
        return first.text[:2500]

    def chunks(self, size: int) -> list[tuple[str, str]]:
        """(section title, slice of text) pairs, each slice at most about `size` characters."""
        return [(section.title, chunk) for section in self.sections for chunk in split_into_chunks(section.text, size)]


def heading_pattern(title: str) -> re.Pattern:
    number, _, words = title.partition(" ")
    is_numbered = bool(re.fullmatch(r"[A-Z0-9][.\d]*", number))
    words = words if is_numbered else title
    words_pattern = r"\s+".join(re.escape(word) for word in clean(words).split())
    number_pattern = rf"{re.escape(number)}\s+" if is_numbered else ""
    return re.compile(rf"(^|\n){number_pattern}{words_pattern}\s*\n")


def sections_from_outline(text: str, outline: list[str]) -> list[Section]:
    """Cuts the text at each outline heading, in order; headings that cannot be found are merged into the previous one."""
    cuts = [(0, "Abstract")]
    position = 0
    for title in outline:
        found = heading_pattern(title).search(text, position)
        if found:
            cuts.append((found.end(), title))
            position = found.end()
    next_starts = [heading_start(text, title, start) for start, title in cuts[1:]] + [len(text)]
    sections = [Section(title, text[start:end].strip()) for (start, title), end in zip(cuts, next_starts)]
    return [section for section in sections if section.text]


def heading_start(text: str, title: str, heading_end: int) -> int:
    """Where a heading begins, so the previous section stops before it."""
    found = heading_pattern(title).search(text, max(0, heading_end - 300), heading_end)
    return found.start() if found else heading_end


def sections_from_markdown(text: str) -> list[Section]:
    parts = re.split(r"^#{1,3}\s+(.+)$", text, flags=re.MULTILINE)
    sections = [Section("Abstract", parts[0].strip())] if parts[0].strip() else []
    sections += [Section(title.strip(), body.strip()) for title, body in zip(parts[1::2], parts[2::2]) if body.strip()]
    return sections


def load_paper(path: str | Path, include_appendix: bool = False) -> Paper:
    path = Path(path)
    if path.suffix.lower() in (".md", ".txt"):
        text = path.read_text(encoding="utf-8")
        title = text.strip().splitlines()[0].lstrip("# ").strip()
        return Paper(title=title, source=str(path), sections=sections_from_markdown(text))

    document = pymupdf.open(path)
    text = clean("\n".join(page.get_text() for page in document))
    outline = [title for level, title, page in document.get_toc()]
    if not include_appendix:
        outline = [title for title in outline if not APPENDIX_HEADING.match(title)]
        body_end = END_OF_BODY.search(text)
        text = text[:body_end.start()] if body_end else text
    title = document.metadata.get("title") or text.splitlines()[0]
    sections = sections_from_outline(text, outline) if outline else [Section("Full text", text)]
    return Paper(title=clean(title), source=str(path), sections=sections)
