from .charts import VERDICT_LABELS

VERDICT_MARKS = {"supported": "✅", "partially_supported": "🟡", "contradicted": "❌", "unverifiable": "❔"}


def cell(text: str, limit: int = 400) -> str:
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def source_links(check: dict) -> str:
    cited = {item["source_id"].strip("[] ") for item in check["evidence"]}
    links = [f"[{source['id']}]({source['url']})" if source["url"] else source["id"]
             for source in check["sources"] if source["id"] in cited]
    return ", ".join(links) or "—"


def evidence_lines(check: dict) -> str:
    return "<br>".join(f"{item['source_id'].strip('[] ')}: “{cell(item['quote'], 220)}”" for item in check["evidence"]) or "—"


def checks_table(checks: list[dict], text_key: str) -> str:
    rows = ["| Id | Claim | Verdict | Why | Verified evidence | Sources |", "|---|---|---|---|---|---|"]
    for check in checks:
        verdict = f"{VERDICT_MARKS[check['verdict']]} {VERDICT_LABELS[check['verdict']]}"
        if check["verdict"] != check["model_verdict"]:
            verdict += f" (agent said {check['model_verdict']}, no verified quote)"
        rows.append(f"| {check['id']} | {cell(check[text_key], 250)} | {verdict} | {cell(check['explanation'], 450)} "
                    f"| {evidence_lines(check)} | {source_links(check)} |")
    return "\n".join(rows)


def issues_table(issues: list[dict], columns: list[tuple[str, str]]) -> str:
    rows = ["| " + " | ".join(title for title, _ in columns) + " |", "|" + "---|" * len(columns)]
    rows += ["| " + " | ".join(cell(issue.get(key, ""), 300) for _, key in columns) + " |" for issue in issues]
    return "\n".join(rows)


def literature_section(literature: dict) -> str:
    queries = "\n".join(f"- `{query['boolean']}` — {query['rationale']}" for query in literature["queries"])
    included = [record for record in literature["records"] if record["included"]]
    records = "\n".join(f"- [{record['title']}]({record['url']}) ({record['year']}) — "
                        f"{cell(next(iter(record['decisions'].values()))['reason'], 200)}" for record in included)
    agreement = literature["screening_agreement"]
    return f"""## 2. Prior work (PRISMA-aligned)

Search strategy (written by the search-strategy agent, run on arXiv, OpenAlex and Semantic Scholar):
{queries}

Screening agreement between the two screeners: Cohen's κ = {agreement['kappa']}, raw agreement {agreement['raw_agreement']:.0%}.
Disagreements were settled by an adjudicator agent.

![PRISMA flow](figures/prisma.png)

Included records:
{records or "- none"}
"""


def build_report(result: dict, run_info: dict) -> str:
    critic = result["critic"]
    structure, language = result["structure"], result["language"]
    conflicts = "\n".join(f"- {item}" for item in critic["conflicts"]) or "- none found"
    gaps = "\n".join(f"- {item}" for item in critic["gaps"]) or "- none found"
    return f"""# Review: {result['paper_title']}

Model `{run_info['model']}` · {run_info['started']} · {run_info['minutes']:.1f} minutes · paper `{run_info['paper']}`

## 1. Summary

{result['summary']}

![Verdicts](figures/verdicts.png)

{literature_section(result['literature'])}
## 3. Stated facts

Each fact was checked by its own agent, started with a fresh context, that searched the web and read sources. A
verdict counts only with a quote that was found, word for word, in a source the agent read; otherwise it is shown
as unverifiable.

{checks_table(result['fact_checks'], 'statement')}

## 4. Findings

Each finding was checked by a fresh agent with the paper's own text, the screened prior work, academic and web
search, and a Python interpreter for recomputing numbers.

{checks_table(result['finding_checks'], 'claim')}

## 5. Structure, coherence and redundancy

{structure['overall']}

{issues_table(structure['issues'], [('Section', 'section'), ('Kind', 'category'), ('Severity', 'severity'), ('Problem', 'problem'), ('Suggestion', 'suggestion')])}

## 6. Grammar and vocabulary

{len(language['issues'])} issues kept of {language['reported']} reported (an issue is kept only if its words are in the paper and the fix changes them).

{issues_table(language['issues'], [('Section', 'section'), ('Kind', 'category'), ('Text', 'quote'), ('Problem', 'problem'), ('Fix', 'suggestion')])}

![Issues by section](figures/issues_by_section.png)

## 7. Quality assurance

| | Facts | Findings |
|---|---|---|
| Verdicts downgraded for lack of a verified quote | {critic['facts']['downgraded_for_missing_evidence']} | {critic['findings']['downgraded_for_missing_evidence']} |
| Quotes rejected (not found in the cited source) | {critic['facts']['rejected_quotes']} | {critic['findings']['rejected_quotes']} |
| Claims checked without reading any source | {critic['facts']['without_sources']} | {critic['findings']['without_sources']} |

Conflicts inside the review:
{conflicts}

Important claims not checked:
{gaps}

## 8. How the run went

![Timeline](figures/timeline.png)

![Tokens](figures/tokens.png)

Every model call and tool call is in `trace.jsonl`; all results are in `results.json`; the protocol used is in `run.json`.
"""
