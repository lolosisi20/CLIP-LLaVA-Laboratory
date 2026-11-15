import re
import unicodedata

QUOTE_CHARACTERS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "−": "-"})


def clean(text: str) -> str:
    """PDF text made readable: ligatures expanded, words split at line ends joined, spaces collapsed."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def normalized(text: str) -> str:
    """For comparing quotes with sources: case, quotes, dashes and whitespace do not matter."""
    text = unicodedata.normalize("NFKC", text).translate(QUOTE_CHARACTERS).lower()
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_quote(source: str, quote: str) -> bool:
    quote = normalized(quote).strip(" .\"'")
    return len(quote) >= 12 and quote in normalized(source)


def split_into_chunks(text: str, size: int, overlap: int = 0) -> list[str]:
    """Paragraph-aligned slices of about `size` characters."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n|(?<=[.!?])\n", text) if part.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) > size:
            chunks.append(current)
            current = current[-overlap:] if overlap else ""
        current = f"{current}\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def looks_degenerate(text: str) -> bool:
    """A local model sometimes loops or dissolves into symbols; such an answer is asked again."""
    words = text.split()
    if len(words) < 30:
        return False
    phrases = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return len(set(phrases)) / len(phrases) < 0.8
