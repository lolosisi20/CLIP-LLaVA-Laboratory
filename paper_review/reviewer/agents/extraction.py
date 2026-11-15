import re
from typing import Literal

from pydantic import BaseModel, Field

from ..config import settings
from ..llm import ask_json, parallel_map
from ..paper import Paper
from ..retrieval import embed
from ..text import contains_quote, looks_degenerate
from ..trace import log_step

DUPLICATE_SIMILARITY = 0.9


class Fact(BaseModel):
    statement: str = Field(description="the fact as one self-contained sentence (name the subject, no 'it' or 'we')")
    quote: str = Field(description="the sentence of the paper that states it, copied exactly")
    category: Literal["hardware", "definition", "prior_work", "history", "numeric", "other"]
    is_about_this_paper: bool = Field(description="true when it describes this paper's own method, experiments or results")
    checkability: Literal[1, 2, 3, 4, 5] = Field(description="5: a specific number or name outside sources state; 1: vague or opinion")


class FactList(BaseModel):
    facts: list[Fact]


class Finding(BaseModel):
    claim: str = Field(description="the result the authors claim, as one self-contained sentence")
    quote: str = Field(description="the sentence of the paper that claims it, copied exactly")
    kind: Literal["speed", "memory", "accuracy", "complexity", "capability", "other"]
    support_in_paper: str = Field(description="the table, figure, theorem or experiment the paper gives as support")


class FindingList(BaseModel):
    findings: list[Finding]


FACTS_SYSTEM = """You extract STATED FACTS from a scientific paper: things the text presents as already true about
the world, independent of this paper's contribution. Examples: hardware specifications, definitions, what prior work
showed or proposed, historical events, numbers about datasets or systems.
Do NOT extract the paper's own results, findings, contributions, proposals, opinions or plans: a sentence about
the proposed method, "we" or "our" experiments is never a stated fact.
Skip trivial statements that carry no checkable information ("attention has a standard implementation").
Only extract facts that are literally in the text; copy the sentence that states each one exactly."""

FINDINGS_SYSTEM = """You extract the FINDINGS of a scientific paper: the results and claims the authors make about
their own work. Examples: speedups, memory savings, accuracy or quality gains, complexity bounds, new capabilities.
Prefer quantitative claims. Do NOT extract background facts about prior work or the world.
Copy the sentence that makes each claim exactly."""


def extract_from_chunk(section: str, chunk: str, schema, system: str, agent: str) -> list[dict]:
    prompt = f"Section: {section}\n---\n{chunk}\n---\nExtract them from this part of the paper. Return an empty list if there are none."
    items = []
    for attempt in range(2):
        answer = ask_json(prompt, schema, agent=agent, system=system, max_tokens=10000, think=True)
        items = [item.model_dump() for item in (answer.facts if isinstance(answer, FactList) else answer.findings if answer else [])]
        if not any(looks_degenerate(str(value)) for item in items for value in item.values()):
            break
    return [{**item, "section": section} for item in items]


def drop_near_duplicates(items: list[dict], key: str) -> list[dict]:
    if len(items) < 2:
        return items
    vectors = embed([item[key] for item in items])
    kept = []
    for position, vector in enumerate(vectors):
        is_duplicate = any(float(vector @ vectors[other]) > DUPLICATE_SIMILARITY for other in kept)
        if not is_duplicate:
            kept.append(position)
    return [items[position] for position in kept]


def extract(paper: Paper, schema, system: str, agent: str, text_key: str) -> list[dict]:
    chunks = paper.chunks(settings.protocol["paper"]["chunk_characters"])
    per_chunk = parallel_map(lambda pair: extract_from_chunk(*pair, schema, system, agent), chunks,
                             settings.protocol["review"]["parallel_agents"])
    items = [item for chunk_items in per_chunk for item in chunk_items]
    grounded = [item for item in items if contains_quote(paper.text, item["quote"])]
    unique = drop_near_duplicates(grounded, text_key)
    log_step(agent, f"{len(items)} extracted, {len(grounded)} quoted from the paper, {len(unique)} after removing duplicates")
    return unique


def method_name(paper: Paper) -> str:
    """The name the paper gives its method: the title's words before the colon (for example "FlashAttention")."""
    head = paper.title.split(":")[0]
    return head.strip().lower() if ":" in paper.title and len(head.split()) <= 3 else ""


def is_stated_fact(fact: dict, name: str) -> bool:
    statement = fact["statement"].lower()
    names_the_method = bool(name) and name in statement
    speaks_as_authors = bool(re.search(r"\b(we|our|us)\b", statement))
    return not fact["is_about_this_paper"] and not names_the_method and not speaks_as_authors


def extract_facts(paper: Paper) -> list[dict]:
    extracted = extract(paper, FactList, FACTS_SYSTEM, "facts", "statement")
    facts = [fact for fact in extracted if is_stated_fact(fact, method_name(paper))]
    log_step("facts", f"{len(facts)} of {len(extracted)} are stated facts rather than the paper's own work")
    has_number = lambda fact: any(character.isdigit() for character in fact["statement"])
    facts.sort(key=lambda fact: (not has_number(fact), -fact["checkability"], fact["category"] == "other"))
    facts = spread_over_sections(facts, settings.protocol["review"]["max_facts"])
    return [{"id": f"F{number}", **fact} for number, fact in enumerate(facts, 1)]


def spread_over_sections(items: list[dict], limit: int) -> list[dict]:
    """Takes the best item of each section in turn, so the checks cover the whole paper."""
    by_section = {}
    for item in items:
        by_section.setdefault(item["section"], []).append(item)
    chosen = []
    while len(chosen) < limit and any(by_section.values()):
        for queue in by_section.values():
            if queue and len(chosen) < limit:
                chosen.append(queue.pop(0))
    return chosen


def extract_findings(paper: Paper) -> list[dict]:
    findings = extract(paper, FindingList, FINDINGS_SYSTEM, "findings", "claim")
    is_headline = lambda finding: finding["section"] == "Abstract"
    is_quantitative = lambda finding: any(character.isdigit() for character in finding["claim"])
    findings.sort(key=lambda finding: (not is_headline(finding), not is_quantitative(finding), finding["kind"] == "other"))
    findings = findings[:settings.protocol["review"]["max_findings"]]
    return [{"id": f"R{number}", **finding} for number, finding in enumerate(findings, 1)]
