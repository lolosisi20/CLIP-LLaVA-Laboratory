from dataclasses import dataclass
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from ..config import settings
from ..literature import SearchQuery, search_arxiv, search_openalex
from ..llm import ask_json, chat_model
from ..retrieval import HybridIndex, embed
from ..sandbox import run_python
from ..text import looks_degenerate, normalized
from ..trace import log_step, traced
from ..web import Ledger, digest_page, page_text, web_search

MIN_RELEVANCE = 0.3  # embedding similarity below which a quote is not about the claim (calibrated on past runs)

FACT_SYSTEM = """You check ONE stated fact from a scientific paper against outside sources.
1. web_search for it (official documentation, datasheets, the cited paper, textbooks, Wikipedia are good sources).
2. read_page on the one to three most authoritative results. Each page comes back as a digest with numbered quotes.
3. search_papers when the fact is about prior research.
Stop as soon as you have quotes that confirm or contradict the fact, or after about four searches that find nothing.
End with a short note: is it supported, partially supported, contradicted or unverifiable, and which source ids show it."""

FINDING_SYSTEM = """You assess ONE finding a paper claims about its own work. You cannot rerun the experiments; you
check whether the claim makes sense:
1. search_paper: do the paper's own tables, figures and numbers support it exactly as stated?
2. run_python: recompute what can be recomputed (a speedup from reported times, a percentage, a complexity bound
   for concrete sizes). Print the results. A speedup is the baseline time divided by the new time: "15% faster"
   means a ratio of 1.15, and "3x faster" a ratio of 3.
3. search_literature and search_papers: is it plausible next to prior work? A big claim (for example "10x faster")
   needs a baseline and a measurement that justify it. web_search and read_page for anything else.
End with a short note: is it supported, partially supported, contradicted or unverifiable, and which source ids show it."""


class Evidence(BaseModel):
    source_id: str = Field(description="the id in brackets, for example S1, P2, L3 or C1")
    quote: str = Field(description="words copied exactly from that source")


class Verdict(BaseModel):
    verdict: Literal["supported", "partially_supported", "contradicted", "unverifiable"]
    confidence: float = Field(description="0 to 1")
    explanation: str = Field(description="2-4 sentences")
    evidence: list[Evidence]


@dataclass
class ValidationContext:
    paper_title: str
    paper_index: HybridIndex
    literature_index: HybridIndex

    def is_reviewed_paper(self, title: str, opening: str) -> bool:
        """A copy of the reviewed paper cannot confirm its own statements."""
        return normalized(self.paper_title)[:50] in normalized(f"{title} {opening}")


def make_tools(ledger: Ledger, context: ValidationContext, claim: str, agent: str, is_finding: bool) -> list:
    @tool("web_search")
    def search_web(query: str) -> str:
        """Search the web. Returns titles, addresses and snippets."""
        if not query.strip():
            return "The query is empty. Write a few search words."
        try:
            hits = web_search(query)
        except Exception as error:
            return f"Search failed ({type(error).__name__}); try other words or another tool."
        return "\n".join(f"- {hit['title']}\n  {hit['url']}\n  {hit['snippet']}" for hit in hits) or "No results."

    @tool("read_page")
    def read_page(url: str) -> str:
        """Read a web page or PDF and get a digest of what it says about the claim, with verbatim quotes."""
        try:
            title, text = page_text(url)
        except Exception as error:
            return f"Could not read {url}: {type(error).__name__}"
        if context.is_reviewed_paper(title, text[:3000]):
            return "This is the reviewed paper itself, which cannot confirm its own statements. Find an independent source."
        digest, dropped = digest_page(title, text, url, claim, agent)
        source = ledger.add("web", title, url, text, digest.summary, digest.quotes)
        quotes = "\n".join(f'  - "{quote}"' for quote in digest.quotes) or "  (no relevant quotes)"
        note = f"\n  ({dropped} quotes were not in the page and were dropped)" if dropped else ""
        return f"[{source.id}] {title}\n{digest.summary}\n{quotes}{note}"

    @tool("search_papers")
    def search_papers(query: str) -> str:
        """Search arXiv and OpenAlex for research papers. Returns titles, years and abstracts."""
        records = []
        for search in (search_openalex, search_arxiv):
            try:
                records += search(SearchQuery(phrases=[query], rationale=""), 4)
            except Exception:
                pass
        lines = []
        independent = [record for record in records if not context.is_reviewed_paper(record.title, "")]
        for record in independent[:6]:
            source = ledger.add("literature", record.title, record.url, record.abstract)
            lines.append(f"[{source.id}] {record.title} ({record.year})\n  {record.abstract[:700]}")
        return "\n".join(lines) or "No papers found."

    @tool("search_paper")
    def search_paper(query: str) -> str:
        """Search the reviewed paper itself, including its tables. Returns the closest passages."""
        lines = []
        for passage in context.paper_index.search(query, k=3):
            source = ledger.add("paper", f"Reviewed paper, {passage.source}", "", passage.text)
            lines.append(f"[{source.id}] {passage.source}\n{passage.text}")
        return "\n\n".join(lines)

    @tool("search_literature")
    def search_literature(query: str) -> str:
        """Search the prior work collected and screened for this review. Returns abstracts."""
        lines = []
        for passage in context.literature_index.search(query, k=4):
            source = ledger.add("literature", passage.title, passage.source, passage.text)
            lines.append(f"[{source.id}] {passage.title}\n  {passage.text[:900]}")
        return "\n".join(lines) or "The collected literature is empty."

    @tool("run_python")
    def python(code: str) -> str:
        """Run Python code (standard library, numpy) and get what it printed. Use print()."""
        output = run_python(code)
        source = ledger.add("python", "Python computation", "", f"{code}\n--- output ---\n{output}")
        return f"[{source.id}]\n{output}"

    if is_finding:
        return [search_paper, python, search_literature, search_papers, search_web, read_page]
    return [search_web, read_page, search_papers]


def verdict_prompt(claim: str, notes: str, ledger: Ledger) -> str:
    return f"""Claim: {claim}

Investigator's notes:
{notes}

Sources read (only these may be cited, and quotes must be copied exactly from them):
{ledger.listing() or "(none)"}

Give the verdict. "supported" or "contradicted" needs at least one quote that shows it; with no such quote the
verdict is "unverifiable". Check every number in the claim: a number the sources state differently makes the claim
"contradicted", and a claim with a number no quote shows is at most "partially_supported". Copy each quote character for character from the sources above, without changing,
joining or completing words; a short exact clause is better than a long paraphrase."""


def is_about(claim: str, quote: str) -> bool:
    claim_vector, quote_vector = embed([claim, quote])
    return float(claim_vector @ quote_vector) >= MIN_RELEVANCE


def checked_evidence(evidence: list[Evidence], ledger: Ledger, claim: str) -> tuple[list[dict], int]:
    """Keeps only evidence whose quote is really in a source the validator read (as Strap's research verifier does)
    and is about the claim, filed under the source that contains it."""
    valid = []
    for item in evidence:
        source = ledger.locate(item.source_id, item.quote)
        if source and is_about(claim, item.quote):
            valid.append({"source_id": source.id, "quote": item.quote})
    return valid, len(evidence) - len(valid)



class Quotes(BaseModel):
    evidence: list[Evidence]


def second_look_for_quotes(claim: str, verdict: Verdict, ledger: Ledger, agent: str) -> list[dict]:
    """A verdict without a verified quote gets one more chance to point at the words that show it."""
    if not ledger.sources:
        return []
    prompt = f"""Claim: {claim}
You judged it "{verdict.verdict}" because: {verdict.explanation}

Sources:
{ledger.listing()}

No quote you gave was found in these sources. Copy one to three short passages, character for character, from the
sources above that show your judgement. Return an empty list if no passage shows it."""
    answer = ask_json(prompt, Quotes, agent=agent, max_tokens=1500)
    return checked_evidence(answer.evidence, ledger, claim)[0] if answer else []


NUDGE = ("You have not read any source yet, so nothing can back a verdict. Call read_page on the most authoritative "
         "result, or search_paper / search_papers, before you conclude.")


def investigate(agent, task: str, ledger: Ledger, config: dict) -> str:
    """Runs the agent; one that stops before reading any source is sent back once to read one."""
    messages = [HumanMessage(content=task)]
    for attempt in range(2):
        try:
            messages = agent.invoke({"messages": messages}, config=config)["messages"]
        except GraphRecursionError:
            return "(the investigation ran out of steps; judge from the sources read)"
        except Exception as error:
            log_step(config["metadata"]["agent"], f"investigation stopped: {type(error).__name__}: {error}")
            return "(the investigation stopped on an error; judge from the sources read)"
        if ledger.sources:
            break
        messages.append(HumanMessage(content=NUDGE))
    return messages[-1].content or "(no closing note)"


def validate(item: dict, context: ValidationContext, is_finding: bool) -> dict:
    claim = item["claim"] if is_finding else item["statement"]
    agent_name = f"validator:{item['id']}"
    ledger = Ledger()
    agent = create_agent(chat_model(8000, is_agent=True), make_tools(ledger, context, claim, agent_name, is_finding),
                         system_prompt=FINDING_SYSTEM if is_finding else FACT_SYSTEM)
    task = f"{'Finding' if is_finding else 'Fact'} from the paper (section {item['section']}):\n{claim}\n\nPaper's sentence: \"{item['quote']}\""
    steps = settings.protocol["review"]["validator_steps"]
    config = {**traced(agent_name), "recursion_limit": 2 * steps + 1}

    notes = investigate(agent, task, ledger, config)
    verdict = None
    for attempt in range(2):
        verdict = ask_json(verdict_prompt(claim, notes, ledger), Verdict, agent=agent_name, max_tokens=2000)
        if verdict and not looks_degenerate(verdict.explanation):
            break
    verdict = verdict or Verdict(verdict="unverifiable", confidence=0, explanation="No verdict could be produced.", evidence=[])
    evidence, rejected = checked_evidence(verdict.evidence, ledger, claim)

    label = verdict.verdict
    is_conclusive = label != "unverifiable"
    if is_conclusive and not evidence:
        evidence = second_look_for_quotes(claim, verdict, ledger, agent_name)
    if is_conclusive and not evidence:
        label = "unverifiable"
    log_step(agent_name, f"{label} ({len(ledger.sources)} sources, {len(evidence)} verified quotes, {rejected} rejected)")
    return {
        **item,
        "verdict": label,
        "model_verdict": verdict.verdict,
        "confidence": verdict.confidence,
        "explanation": verdict.explanation,
        "evidence": evidence,
        "rejected_evidence": rejected,
        "notes": notes,
        "sources": [source.public() for source in ledger.sources],
    }
