import os
import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .config import settings
from .http import cached_get
from .llm import ask_json, parallel_map
from .trace import log_step

ATOM = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class Record:
    title: str
    abstract: str
    year: int | None
    url: str
    sources: list[str] = field(default_factory=list)
    doi: str = ""
    included: bool | None = None
    decisions: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.doi.lower() or re.sub(r"[^a-z0-9]", "", self.title.lower())[:80]


# ── Search strategy ──

class SearchQuery(BaseModel):
    phrases: list[str] = Field(description="2 or 3 key phrases of one to three words each, for example 'attention' or 'GPU memory'")
    rationale: str


class SearchPlan(BaseModel):
    queries: list[SearchQuery]


def plan_search(title: str, abstract: str, count: int) -> list[SearchQuery]:
    prompt = f"""You design the literature search for reviewing this paper. Write {count} search queries that would
find the prior work its claims build on or compete with: the problem, the methods, the baselines and the hardware.
Each query is a list of 2 or 3 short key phrases (one to three words each, as an index would list them); they are
combined with AND, so a long phrase finds nothing. Make the queries differ from each other.

Title: {title}
Abstract: {abstract}"""
    plan = ask_json(prompt, SearchPlan, agent="search_strategy")
    return plan.queries[:count] if plan else [SearchQuery(phrases=[title], rationale="fallback: the title")]


def keyword_query(query: SearchQuery) -> str:
    return " ".join(query.phrases)


def boolean_query(query: SearchQuery) -> str:
    return " AND ".join(f'"{phrase}"' for phrase in query.phrases)


# ── Sources ──

def search_arxiv(query: SearchQuery, limit: int) -> list[Record]:
    search = " AND ".join(f'all:"{phrase}"' for phrase in query.phrases)
    xml = cached_get("https://export.arxiv.org/api/query", {"search_query": search, "max_results": limit})
    records = []
    for entry in ElementTree.fromstring(xml).findall("atom:entry", ATOM):
        records.append(Record(
            title=" ".join(entry.findtext("atom:title", "", ATOM).split()),
            abstract=" ".join(entry.findtext("atom:summary", "", ATOM).split()),
            year=int(entry.findtext("atom:published", "0", ATOM)[:4]) or None,
            url=entry.findtext("atom:id", "", ATOM),
            sources=["arxiv"],
        ))
    return records


def abstract_from_inverted_index(index: dict | None) -> str:
    if not index:
        return ""
    positions = sorted((position, word) for word, places in index.items() for position in places)
    return " ".join(word for _, word in positions)


def search_openalex(query: SearchQuery, limit: int) -> list[Record]:
    key = os.environ.get("OPENALEX_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else None
    data = cached_get("https://api.openalex.org/works", {"search": keyword_query(query), "per-page": limit}, as_json=True, headers=headers)
    return [Record(
        title=work.get("title") or "",
        abstract=abstract_from_inverted_index(work.get("abstract_inverted_index")),
        year=work.get("publication_year"),
        url=work.get("doi") or work.get("id"),
        doi=(work.get("doi") or "").removeprefix("https://doi.org/"),
        sources=["openalex"],
    ) for work in data.get("results", [])]


def search_semantic_scholar(query: SearchQuery, limit: int) -> list[Record]:
    params = {"query": keyword_query(query), "limit": limit, "fields": "title,abstract,year,url,externalIds"}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"x-api-key": key} if key else None
    data = cached_get("https://api.semanticscholar.org/graph/v1/paper/search", params, as_json=True, retries=4, headers=headers)
    return [Record(
        title=paper.get("title") or "",
        abstract=paper.get("abstract") or "",
        year=paper.get("year"),
        url=paper.get("url") or "",
        doi=(paper.get("externalIds") or {}).get("DOI", "") or "",
        sources=["semantic_scholar"],
    ) for paper in data.get("data") or []]


SOURCES = {"arxiv": search_arxiv, "openalex": search_openalex, "semantic_scholar": search_semantic_scholar}


def retrieve(queries: list[SearchQuery], source_names: list[str], limit: int) -> dict[str, list[Record]]:
    found = {name: [] for name in source_names}
    for query in queries:
        for name in source_names:
            try:
                found[name] += SOURCES[name](query, limit)
            except Exception as error:
                log_step("retrieval", f"{name} failed for {keyword_query(query)!r}: {error}")
    return found


def deduplicate(records: list[Record]) -> list[Record]:
    unique: dict[str, Record] = {}
    for record in records:
        kept = unique.get(record.key)
        if kept is None:
            unique[record.key] = record
            continue
        kept.sources = sorted(set(kept.sources + record.sources))
        kept.abstract = kept.abstract or record.abstract
        kept.doi = kept.doi or record.doi
    return list(unique.values())


# ── Screening (two independent screeners, a third resolves disagreements) ──

class ScreeningDecision(BaseModel):
    include: bool
    reason: str = Field(description="one sentence, naming the criterion that decided it")


SCREENERS = {
    "screener_a": "You are a strict systematic-review screener. Include only records that clearly meet the criteria.",
    "screener_b": "You are a systematic-review screener who reads the abstract first and the title last, "
                  "and judges whether the record could confirm or contradict the reviewed paper's claims.",
}


def screening_prompt(record: Record, paper_title: str, criteria: str) -> str:
    return f"""Reviewed paper: {paper_title}

Criteria:{criteria}
Record to screen:
Title: {record.title}
Year: {record.year}
Abstract: {record.abstract[:2500] or "(no abstract)"}

Decide whether this record is included."""


def screen_record(record: Record, paper_title: str, criteria: str) -> Record:
    prompt = screening_prompt(record, paper_title, criteria)
    for name, system in SCREENERS.items():
        decision = ask_json(prompt, ScreeningDecision, agent=name, system=system, max_tokens=1000)
        record.decisions[name] = decision.model_dump() if decision else {"include": False, "reason": "no answer"}

    votes = [decision["include"] for decision in record.decisions.values()]
    if len(set(votes)) == 1:
        record.included = votes[0]
        return record

    reasons = "\n".join(f"- {name}: {'include' if d['include'] else 'exclude'} because {d['reason']}"
                        for name, d in record.decisions.items())
    verdict = ask_json(f"{prompt}\n\nTwo screeners disagreed:\n{reasons}\nMake the final decision.",
                       ScreeningDecision, agent="screening_adjudicator", max_tokens=1000)
    record.decisions["adjudicator"] = verdict.model_dump() if verdict else {"include": False, "reason": "no answer"}
    record.included = record.decisions["adjudicator"]["include"]
    return record


def cohen_kappa(first: list[bool], second: list[bool]) -> float:
    if not first:
        return float("nan")
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    p_first, p_second = sum(first) / len(first), sum(second) / len(second)
    expected = p_first * p_second + (1 - p_first) * (1 - p_second)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def is_same_paper(record: Record, paper_title: str) -> bool:
    simplify = lambda text: re.sub(r"[^a-z0-9]", "", text.lower())
    return simplify(paper_title)[:40] in simplify(record.title) or simplify(record.title)[:40] in simplify(paper_title)


def run_literature_review(paper_title: str, abstract: str) -> dict:
    protocol = settings.protocol["literature"]
    queries = plan_search(paper_title, abstract, protocol["queries"])
    log_step("search_strategy", f"{len(queries)} queries: " + "; ".join(boolean_query(query) for query in queries))

    found = retrieve(queries, protocol["sources"], protocol["results_per_query"])
    identified = [record for records in found.values() for record in records]
    unique = deduplicate(identified)
    itself = [record for record in unique if is_same_paper(record, paper_title)]
    no_abstract = [record for record in unique if record not in itself and not record.abstract]
    to_screen = [record for record in unique if record not in itself and record.abstract]
    log_step("retrieval", f"{len(identified)} records, {len(unique)} unique, {len(to_screen)} to screen")

    screened = parallel_map(lambda record: screen_record(record, paper_title, protocol["screening_criteria"]),
                            to_screen, settings.protocol["review"]["parallel_agents"])
    included = [record for record in screened if record.included]
    log_step("screening", f"{len(included)} of {len(screened)} records included")

    votes_a = [record.decisions["screener_a"]["include"] for record in screened]
    votes_b = [record.decisions["screener_b"]["include"] for record in screened]
    return {
        "queries": [{"phrases": q.phrases, "boolean": boolean_query(q), "rationale": q.rationale} for q in queries],
        "prisma": {
            "identified": {name: len(records) for name, records in found.items()},
            "duplicates_removed": len(identified) - len(unique),
            "removed_before_screening": {"the reviewed paper itself": len(itself), "no abstract": len(no_abstract)},
            "screened": len(screened),
            "excluded": len(screened) - len(included),
            "included": len(included),
        },
        "screening_agreement": {
            "kappa": round(cohen_kappa(votes_a, votes_b), 3),
            "raw_agreement": round(sum(a == b for a, b in zip(votes_a, votes_b)) / max(1, len(screened)), 3),
        },
        "records": screened,
    }
