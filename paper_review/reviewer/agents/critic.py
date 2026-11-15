from collections import Counter

from pydantic import BaseModel, Field

from ..llm import ask, ask_json
from ..trace import log_step


class CrossCheck(BaseModel):
    conflicts: list[str] = Field(description="places where the review's own conclusions disagree or a finding rests on a fact found false")
    gaps: list[str] = Field(description="important claims of the paper that the review did not check")


def audit(checks: list[dict]) -> dict:
    """Counts the review must be transparent about: verdicts downgraded and quotes rejected by the quote check."""
    return {
        "verdicts": dict(Counter(check["verdict"] for check in checks)),
        "downgraded_for_missing_evidence": sum(check["verdict"] != check["model_verdict"] for check in checks),
        "rejected_quotes": sum(check["rejected_evidence"] for check in checks),
        "without_sources": sum(not check["sources"] for check in checks),
    }


def one_line(check: dict) -> str:
    text = check.get("statement") or check.get("claim")
    return f"[{check['id']}] {check['verdict']}: {text} — {check['explanation'][:300]}"


def review_the_review(paper_abstract: str, fact_checks: list[dict], finding_checks: list[dict], structure: dict) -> dict:
    summary = "\n".join(one_line(check) for check in fact_checks + finding_checks)
    prompt = f"""You are the quality-assurance agent of a paper review. Read the paper's abstract and the review's
conclusions, and find (1) conflicts inside the review, for example a finding judged supported although a fact it
depends on was contradicted, and (2) important claims of the abstract the review never checked.

Abstract:
{paper_abstract}

Review conclusions:
{summary}

Structure assessment: {structure.get('overall', '')}"""
    cross_check = ask_json(prompt, CrossCheck, agent="critic", max_tokens=2000)
    result = {
        "facts": audit(fact_checks),
        "findings": audit(finding_checks),
        "conflicts": cross_check.conflicts if cross_check else [],
        "gaps": cross_check.gaps if cross_check else [],
    }
    log_step("critic", f"{len(result['conflicts'])} conflicts, {len(result['gaps'])} gaps")
    return result


def write_summary(title: str, fact_checks: list[dict], finding_checks: list[dict], structure: dict,
                  language: dict, critic: dict) -> str:
    language_counts = Counter(issue["category"] for issue in language["issues"])
    prompt = f"""Write the executive summary of a review of the paper "{title}" for its editor, in Markdown:
a short overall assessment, then "Strengths" and "Weaknesses" as bullet lists, then a one-line recommendation
(accept, minor revision, major revision or reject) with its reason. Use only what is below; cite claim ids like [F3].

Facts checked:
{chr(10).join(one_line(check) for check in fact_checks)}

Findings checked:
{chr(10).join(one_line(check) for check in finding_checks)}

Structure: {structure.get('overall', '')}
Structure issues: {len(structure['issues'])} ({sum(issue['severity'] == 'major' for issue in structure['issues'])} major)
Language issues: {dict(language_counts)}
Quality assurance: conflicts {critic['conflicts']}; unchecked claims {critic['gaps']}"""
    return ask(prompt, agent="synthesis", max_tokens=2500)
