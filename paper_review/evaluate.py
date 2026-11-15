"""Evaluation of the pipeline.

  python evaluate.py planted  papers/flashattention.pdf      plants known errors in a copy of the paper, reviews it, scores what was caught
  python evaluate.py agreement runs/<run> --model NAME       re-validates a run's claims with another model (or the same one) and measures agreement
  python evaluate.py time runs/<run>                         compares the run's time with an estimate of the same review done by hand
"""
import argparse
import json
import re
import sys
from pathlib import Path

from reviewer.agents.validator import ValidationContext, validate
from reviewer.charts import SERIES, INK_SECONDARY, plt, save
from reviewer.config import ROOT, settings
from reviewer.literature import cohen_kappa
from reviewer.llm import parallel_map
from reviewer.paper import load_paper
from reviewer.retrieval import HybridIndex, Passage
from reviewer.text import normalized

EVALUATION = ROOT / "evaluation"
FLAGGED = {"contradicted", "partially_supported"}

# (kind, original text, planted text). Each planted text is unique in the paper, so a catch can be traced to it.
PLANTED = [
    ("fact", "the A100 GPU has 40-80GB of high bandwidth memory", "the A100 GPU has 4-8GB of high bandwidth memory"),
    ("fact", "which is the number of arithmetic operations per byte of memory access",
             "which is the number of arithmetic operations per second of compute time"),
    ("fact", "The on-chip SRAM is an order of magnitude faster than HBM", "The on-chip SRAM is an order of magnitude slower than HBM"),
    ("finding", "3× speedup on GPT-2 (seq. length 1K)", "30× speedup on GPT-2 (seq. length 1K)"),
    ("finding", "15% end-to-end wall-clock speedup on BERT-large", "85% end-to-end wall-clock speedup on BERT-large"),
    ("grammar", "Transformers have grown larger", "Transformers has grown larger"),
    ("grammar", "to make it easier to build on this primitive", "to make it more easier to build on this primitive"),
    ("grammar", "We discuss limitations of our approach", "We discusses limitation of our approach"),
    ("grammar", "most operations in Transformers are bottlenecked by memory accesses",
                "most operations in Transformers are bottlenecked by memory acceses"),
]


# ── Planted errors ──

def flexible(text: str) -> re.Pattern:
    """Matches the words of `text` across the line breaks of PDF text."""
    return re.compile(r"\s+".join(re.escape(word) for word in text.split()))


def planted_copy(pdf: Path) -> Path:
    paper = load_paper(pdf)
    sections = {section.title: section.text for section in paper.sections}
    hardware = next(title for title in sections if title.startswith("2.1"))
    duplicate = re.search(r"GPU Memory Hierarchy\..*?(?=Execution Model\.)", sections[hardware], re.DOTALL).group(0)

    for kind, original, planted in PLANTED:
        title = next(title for title, text in sections.items() if flexible(original).search(text))
        sections[title] = flexible(original).sub(planted, sections[title], count=1)
    benchmark = next(title for title in sections if title.startswith("4.3"))
    sections[benchmark] = duplicate.strip() + "\n\n" + sections[benchmark]

    path = EVALUATION / f"{pdf.stem}_planted.md"
    body = "\n\n".join(f"# {title}\n{text}" for title, text in sections.items())
    path.write_text(f"# {paper.title}\n\n{body}", encoding="utf-8")
    return path


def shows_plant(quote: str, original: str, planted: str) -> bool:
    """The quote is taken from the planted sentence and contains the words that were changed."""
    quote, planted_text = normalized(quote), normalized(planted)
    changed = [normalized(word) for word in planted.split() if word not in original.split()]
    from_planted = planted_text[:25] in quote or quote in planted_text
    return from_planted and any(word in quote for word in changed)


def score_planted(result: dict) -> list[dict]:
    rows = []
    for kind, original, planted in PLANTED:
        if kind in ("fact", "finding"):
            checks = result["fact_checks"] if kind == "fact" else result["finding_checks"]
            matching = [check for check in checks if shows_plant(check["quote"], original, planted)]
            extracted = bool(matching)
            caught = any(check["verdict"] in FLAGGED for check in matching)
        else:
            extracted = True
            caught = any(shows_plant(issue["quote"], original, planted) for issue in result["language"]["issues"])
        rows.append({"kind": kind, "planted": planted, "extracted": extracted, "caught": caught})

    redundancy = [issue for issue in result["structure"]["issues"] if issue["category"] == "redundancy"]
    caught = any("4.3" in issue["section"] and "2.1" in issue["section"] for issue in redundancy)
    rows.append({"kind": "redundancy", "planted": "paragraph of 2.1 copied into 4.3", "extracted": True, "caught": caught})
    return rows


KINDS = {"fact": "false facts", "finding": "inflated findings", "grammar": "grammar errors", "redundancy": "duplicated paragraph"}


def short_name(model: str) -> str:
    return model.split("-")[0] + " " + next((part for part in model.split("-") if part.endswith("b")), "")


def planted_chart(results: dict[str, list[dict]], path: Path):
    """Caught against planted, per kind of error, one bar per model."""
    figure, axes = plt.subplots(figsize=(8, 3.6))
    height = 0.8 / len(results)
    for index, (model, rows) in enumerate(results.items()):
        caught = [sum(row["kind"] == kind and row["caught"] for row in rows) for kind in KINDS]
        positions = [position + index * height for position in range(len(KINDS))]
        axes.barh(positions, caught, height=height * 0.9, color=SERIES[index], label=short_name(model))
        for position, count in zip(positions, caught):
            axes.text(count + 0.05, position, str(count), va="center", color=INK_SECONDARY, fontsize=9)
    planted = [sum(row["kind"] == kind for row in next(iter(results.values()))) for kind in KINDS]
    axes.set_yticks([position + 0.4 - height / 2 for position in range(len(KINDS))],
                    [f"{label} ({total} planted)" for label, total in zip(KINDS.values(), planted)])
    axes.invert_yaxis()
    axes.set_xlim(0, max(planted) + 0.5)
    axes.set_xticks(range(max(planted) + 1))
    axes.set_xlabel("caught")
    axes.spines["left"].set_visible(False)
    axes.tick_params(axis="y", length=0)
    if len(results) > 1:
        axes.legend(frameon=False, ncol=len(results), loc="lower left", bbox_to_anchor=(0, 1.0))
    axes.set_title("Planted errors caught", pad=26 if len(results) > 1 else 8)
    save(figure, path)


def planted_results() -> dict[str, list[dict]]:
    files = sorted(EVALUATION.glob("planted_*.json"), key=lambda path: "27b" in path.name)
    return {json.loads(path.read_text(encoding="utf-8"))["model"]: json.loads(path.read_text(encoding="utf-8"))["rows"] for path in files}


def run_planted(pdf: Path):
    import review
    copy = planted_copy(pdf)
    sys.argv = ["review.py", str(copy), "--model", settings.model]
    review.main()
    run_dir = max((ROOT / "runs").glob(f"{copy.stem}_{settings.model.replace('.', '-')}_*"), key=lambda path: path.stat().st_mtime)
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    rows = score_planted(result)
    summary = {"run": run_dir.name, "model": settings.model, "rows": rows}
    (EVALUATION / f"planted_{settings.model}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    planted_chart(planted_results(), EVALUATION / "planted.png")
    print(f"caught {sum(row['caught'] for row in rows)} of {len(rows)} planted errors ({run_dir.name})")


# ── Agreement between two validations of the same claims ──

def run_agreement(run_dir: Path):
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    paper = load_paper(ROOT / "papers" / json.loads((run_dir / "run.json").read_text())["paper"])
    included = [record for record in result["literature"]["records"] if record["included"]]
    context = ValidationContext(
        paper_title=paper.title,
        paper_index=HybridIndex([Passage(chunk, section) for section, chunk in paper.chunks(1200)]),
        literature_index=HybridIndex([Passage(record["abstract"], record["url"], record["title"]) for record in included]),
    )
    workers = settings.protocol["review"]["parallel_agents"]
    facts = parallel_map(lambda check: validate(strip_check(check), context, is_finding=False), result["fact_checks"], workers)
    findings = parallel_map(lambda check: validate(strip_check(check), context, is_finding=True), result["finding_checks"], workers)

    first = result["fact_checks"] + result["finding_checks"]
    second = facts + findings
    pairs = [{"id": a["id"], "first": a["verdict"], "second": b["verdict"]} for a, b in zip(first, second)]
    agreement = {
        "run": run_dir.name, "first_model": json.loads((run_dir / "run.json").read_text())["model"], "second_model": settings.model,
        "exact_agreement": round(sum(pair["first"] == pair["second"] for pair in pairs) / len(pairs), 3),
        "flagged_kappa": round(cohen_kappa([pair["first"] in FLAGGED for pair in pairs], [pair["second"] in FLAGGED for pair in pairs]), 3),
        "supported_kappa": round(cohen_kappa([pair["first"] == "supported" for pair in pairs], [pair["second"] == "supported" for pair in pairs]), 3),
        "pairs": pairs, "second_checks": second,
    }
    name = f"agreement_{settings.model}.json"
    (EVALUATION / name).write_text(json.dumps(agreement, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"exact agreement {agreement['exact_agreement']:.0%}, kappa (supported or not) {agreement['supported_kappa']}")


def strip_check(check: dict) -> dict:
    keep = ("id", "section", "statement", "claim", "quote", "category", "kind", "checkability", "support_in_paper")
    return {key: check[key] for key in keep if key in check}


# ── Time against a review by hand ──

MANUAL_MINUTES = {  # assumptions for one experienced reviewer, written down so they can be argued with
    "search strategy": 30,
    "per record screened": 1.5,
    "per fact checked": 6,
    "per finding checked": 15,
    "structure review": 45,
    "per 1000 words copy-edited": 8,
    "writing the report": 60,
}


def run_time(run_dir: Path):
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    paper = load_paper(ROOT / "papers" / run["paper"])
    words = len(paper.text.split())
    manual = {
        "literature search and screening": MANUAL_MINUTES["search strategy"] + MANUAL_MINUTES["per record screened"] * result["literature"]["prisma"]["screened"],
        "fact checking": MANUAL_MINUTES["per fact checked"] * len(result["fact_checks"]),
        "finding checking": MANUAL_MINUTES["per finding checked"] * len(result["finding_checks"]),
        "structure review": MANUAL_MINUTES["structure review"],
        "copy-editing": MANUAL_MINUTES["per 1000 words copy-edited"] * words / 1000,
        "writing the report": MANUAL_MINUTES["writing the report"],
    }
    total = sum(manual.values())
    figure, axes = plt.subplots(figsize=(8, 2.4))
    axes.barh([0, 1], [total / 60, run["minutes"] / 60], color=["#c3c2b7", SERIES[0]], height=0.55)
    for position, hours in enumerate([total / 60, run["minutes"] / 60]):
        axes.text(hours, position, f" {hours:.1f} h", va="center", color=INK_SECONDARY)
    axes.set_yticks([0, 1], ["by hand (estimate)", f"pipeline ({run['model'].split('-')[0]})"])
    axes.invert_yaxis()
    axes.set_xlabel("hours")
    axes.spines["left"].set_visible(False)
    axes.tick_params(axis="y", length=0)
    axes.set_title("Time for the same review", pad=8)
    save(figure, EVALUATION / "time.png")
    summary = {"run": run_dir.name, "pipeline_minutes": round(run["minutes"], 1), "manual_minutes": round(total),
               "manual_breakdown": {key: round(value) for key, value in manual.items()}, "assumptions": MANUAL_MINUTES,
               "speedup": round(total / run["minutes"], 1)}
    (EVALUATION / "time.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"pipeline {run['minutes']:.0f} min against about {total:.0f} min by hand ({summary['speedup']}x)")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["planted", "agreement", "time"])
    parser.add_argument("target", help="a paper (planted) or a run folder (agreement, time)")
    parser.add_argument("--model", default=settings.model)
    arguments = parser.parse_args()
    settings.model = arguments.model
    EVALUATION.mkdir(exist_ok=True)
    {"planted": run_planted, "agreement": run_agreement, "time": run_time}[arguments.mode](Path(arguments.target))


if __name__ == "__main__":
    main()
