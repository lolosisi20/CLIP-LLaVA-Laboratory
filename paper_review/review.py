"""Reviews a paper end to end: python review.py papers/flashattention.pdf [--model NAME] [--human]"""
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from langgraph.types import Command

from reviewer.charts import draw_all
from reviewer.config import ROOT, read_protocol, settings
from reviewer.graph import build_review_graph
from reviewer.report import build_report
from reviewer.trace import log_step, start_trace


def parse_arguments():
    parser = argparse.ArgumentParser(description="Agentic paper review")
    parser.add_argument("paper", help="a PDF, or a Markdown/text file with # headings")
    parser.add_argument("--model", default=settings.model, help="model id on the OpenAI-compatible server")
    parser.add_argument("--human", action="store_true", help="pause after fact extraction to drop facts")
    parser.add_argument("--protocol", default=str(ROOT / "protocol.toml"), help="the review protocol (TOML)")
    parser.add_argument("--runs", default=str(ROOT / "runs"), help="folder for the run outputs")
    return parser.parse_args()


def new_run_dir(paper: Path, model: str, runs: Path) -> Path:
    model_name = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    run_dir = runs / f"{paper.stem}_{model_name}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    return run_dir


def ask_human(question: dict) -> list[str]:
    print(f"\n{question['question']}")
    for fact in question["facts"]:
        print(f"  {fact['id']}: {fact['statement']}")
    answer = input("Ids to drop, separated by spaces (Enter keeps all): ")
    return answer.upper().split()


def run_graph(paper: Path, run_id: str) -> dict:
    graph = build_review_graph()
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": 50}
    payload = {"paper_path": str(paper)}
    while True:
        state = graph.invoke(payload, config)
        pending = state.get("__interrupt__")
        if not pending:
            return state
        payload = Command(resume=ask_human(pending[0].value))


def git_commit() -> str:
    finished = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    return finished.stdout.strip()


def to_result(state: dict) -> dict:
    keys = ("literature", "facts", "findings", "fact_checks", "finding_checks", "structure", "language", "critic", "summary")
    return {"paper_title": state["paper"].title, **{key: state[key] for key in keys}}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    arguments = parse_arguments()
    settings.model = arguments.model
    settings.protocol = read_protocol(arguments.protocol)
    settings.protocol["review"]["human_review"] |= arguments.human
    paper = Path(arguments.paper).resolve()
    run_dir = new_run_dir(paper, settings.model, Path(arguments.runs))
    start_trace(run_dir / "trace.jsonl")

    run_info = {"paper": paper.name, "model": settings.model, "profile": settings.profile,
                "started": f"{datetime.now():%Y-%m-%d %H:%M}", "commit": git_commit(), "protocol": settings.protocol}
    log_step("pipeline", f"reviewing {paper.name} with {settings.model}; output in {run_dir}")
    began = time.time()
    result = to_result(run_graph(paper, run_dir.name))
    run_info["minutes"] = (time.time() - began) / 60

    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    draw_all(result, run_dir)
    (run_dir / "report.md").write_text(build_report(result, run_info), encoding="utf-8")
    log_step("pipeline", f"done in {run_info['minutes']:.1f} minutes: {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
