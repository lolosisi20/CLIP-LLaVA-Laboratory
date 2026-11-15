from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from ..config import settings
from ..llm import ask_json, parallel_map
from ..paper import Paper
from ..retrieval import embed
from ..text import split_into_chunks
from ..trace import log_step

REPEAT_SIMILARITY = 0.85
MAX_PAIRS = 8


class StructureIssue(BaseModel):
    section: str
    category: Literal["structure", "coherence", "flow", "balance", "missing"]
    problem: str = Field(description="what is wrong, in one or two full sentences that point at the passage")
    severity: Literal["minor", "major"]
    suggestion: str = Field(description="the concrete change that fixes it")


class StructureReview(BaseModel):
    overall: str = Field(description="3-5 sentences on how the paper is organised and how well it argues")
    issues: list[StructureIssue]


class RepeatJudgement(BaseModel):
    is_redundant: bool
    explanation: str


STRUCTURE_SYSTEM = """You review the STRUCTURE and COHERENCE of a scientific paper, not its grammar and not whether
its results are true. Check: the order of sections, whether the introduction's promises are delivered, whether terms
are defined before use, transitions between sections, sections that are too thin or too long, and missing parts a
reader would expect (limitations, related work, reproducibility details). Be specific: name the section.
Categories: structure (order and placement of parts), coherence (claims or terms that do not fit together),
flow (missing transitions, jumps), balance (a part too thin or too long for its role), missing (an expected part
is absent). Report only problems a careful editor would raise; a well-organised paper may have very few."""


def review_outline(paper: Paper) -> StructureReview | None:
    outline = "\n".join(f"- {section.title} ({len(section.text)} characters)" for section in paper.sections)
    prompt = f"Outline:\n{outline}\n\nFull text:\n{paper.text}"
    return ask_json(prompt, StructureReview, agent="structure", system=STRUCTURE_SYSTEM, max_tokens=3000)


def repeated_passages(paper: Paper) -> list[tuple[dict, dict, float]]:
    """Pairs of paragraphs in different sections that say nearly the same thing."""
    paragraphs = [{"section": section.title, "text": chunk}
                  for section in paper.sections for chunk in split_into_chunks(section.text, 700) if len(chunk) > 200]
    vectors = embed([paragraph["text"] for paragraph in paragraphs])
    similarity = vectors @ vectors.T
    pairs = []
    for first, second in zip(*np.triu_indices(len(paragraphs), k=1)):
        is_other_section = paragraphs[first]["section"] != paragraphs[second]["section"]
        if is_other_section and similarity[first, second] > REPEAT_SIMILARITY:
            pairs.append((paragraphs[first], paragraphs[second], float(similarity[first, second])))
    return sorted(pairs, key=lambda pair: -pair[2])[:MAX_PAIRS]


def judge_repeat(pair: tuple[dict, dict, float]) -> dict | None:
    first, second, similarity = pair
    prompt = f"""Two passages of the same paper:

[{first['section']}]
{first['text']}

[{second['section']}]
{second['text']}

Is the second one redundant, repeating the first without adding anything? A short summary of the contributions in the
abstract and introduction is expected and not redundant; repeating explanations, numbers or arguments is."""
    judgement = ask_json(prompt, RepeatJudgement, agent="redundancy", max_tokens=800)
    if not judgement or not judgement.is_redundant:
        return None
    return {"section": f"{first['section']} / {second['section']}", "problem": judgement.explanation,
            "category": "redundancy", "severity": "minor", "similarity": round(similarity, 3),
            "suggestion": "Keep the fuller passage and refer to it from the other.",
            "passages": [first["text"][:400], second["text"][:400]]}


def review_structure(paper: Paper) -> dict:
    outline_review = review_outline(paper)
    pairs = repeated_passages(paper)
    repeats = [issue for issue in parallel_map(judge_repeat, pairs, settings.protocol["review"]["parallel_agents"]) if issue]
    issues = [issue.model_dump() for issue in outline_review.issues] if outline_review else []
    log_step("structure", f"{len(issues)} structure issues, {len(repeats)} of {len(pairs)} similar passage pairs judged redundant")
    return {"overall": outline_review.overall if outline_review else "", "issues": issues + repeats}
