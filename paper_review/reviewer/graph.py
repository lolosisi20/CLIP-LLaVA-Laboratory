from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .agents.critic import review_the_review, write_summary
from .agents.extraction import extract_facts, extract_findings
from .agents.language import review_language
from .agents.structure import review_structure
from .agents.validator import ValidationContext, validate as validate_claim
from .config import settings
from .literature import run_literature_review
from .llm import parallel_map
from .paper import Paper, load_paper
from .retrieval import HybridIndex, Passage
from .trace import log_step


class ReviewState(TypedDict, total=False):
    paper_path: str
    paper: Paper
    literature: dict
    facts: list[dict]
    findings: list[dict]
    fact_checks: list[dict]
    finding_checks: list[dict]
    structure: dict
    language: dict
    critic: dict
    summary: str


class Indexes:
    """Search indexes the validators share; built once per run and kept out of the graph state."""
    paper: HybridIndex | None = None
    literature: HybridIndex | None = None


# ── Nodes ──

def ingest(state: ReviewState) -> dict:
    paper = load_paper(state["paper_path"], settings.protocol["paper"]["include_appendix"])
    Indexes.paper = HybridIndex([Passage(chunk, section) for section, chunk in paper.chunks(1200)])
    log_step("ingest", f"{paper.title}: {len(paper.sections)} sections, {len(paper.text)} characters")
    return {"paper": paper}


def literature(state: ReviewState) -> dict:
    review = run_literature_review(state["paper"].title, state["paper"].abstract)
    included = [record for record in review["records"] if record.included]
    Indexes.literature = HybridIndex([Passage(record.abstract, record.url, record.title) for record in included])
    review["records"] = [vars(record) for record in review["records"]]
    return {"literature": review}


def facts(state: ReviewState) -> dict:
    return {"facts": extract_facts(state["paper"])}


def human_review(state: ReviewState) -> dict:
    is_enabled = settings.protocol["review"]["human_review"]
    if not is_enabled:
        return {}
    dropped = interrupt({"question": "Which facts should not be validated?", "facts": state["facts"]})
    kept = [fact for fact in state["facts"] if fact["id"] not in set(dropped or [])]
    log_step("human_review", f"kept {len(kept)} of {len(state['facts'])} facts")
    return {"facts": kept}


def findings(state: ReviewState) -> dict:
    return {"findings": extract_findings(state["paper"])}


def validation_context(paper: Paper) -> ValidationContext:
    return ValidationContext(paper_title=paper.title, paper_index=Indexes.paper,
                             literature_index=Indexes.literature or HybridIndex([]))


def validate(state: ReviewState) -> dict:
    """Facts and findings share one pool of fresh agents, so every model slot stays busy."""
    context = validation_context(state["paper"])
    jobs = [(fact, False) for fact in state["facts"]] + [(finding, True) for finding in state["findings"]]
    checks = parallel_map(lambda job: validate_claim(job[0], context, is_finding=job[1]), jobs,
                          settings.protocol["review"]["parallel_agents"])
    return {"fact_checks": checks[:len(state["facts"])], "finding_checks": checks[len(state["facts"]):]}


def structure(state: ReviewState) -> dict:
    return {"structure": review_structure(state["paper"])}


def language(state: ReviewState) -> dict:
    return {"language": review_language(state["paper"])}


def critic(state: ReviewState) -> dict:
    return {"critic": review_the_review(state["paper"].abstract, state["fact_checks"], state["finding_checks"], state["structure"])}


def synthesis(state: ReviewState) -> dict:
    summary = write_summary(state["paper"].title, state["fact_checks"], state["finding_checks"],
                            state["structure"], state["language"], state["critic"])
    return {"summary": summary}


# ── Wiring ──

def build_review_graph():
    graph = StateGraph(ReviewState)
    for node in (ingest, literature, facts, human_review, findings, validate, structure, language, critic, synthesis):
        graph.add_node(node.__name__, node)

    graph.add_edge(START, "ingest")
    for branch in ("literature", "facts", "findings", "structure", "language"):
        graph.add_edge("ingest", branch)
    graph.add_edge("facts", "human_review")
    graph.add_edge(["human_review", "findings", "literature"], "validate")
    graph.add_edge(["validate", "structure", "language"], "critic")
    graph.add_edge("critic", "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile(checkpointer=MemorySaver())
