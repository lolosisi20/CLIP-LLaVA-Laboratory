"""Builds the explainer video from a finished run: python video/build_video.py runs/<run>

The slides are an HTML page whose animations are all CSS; each frame is taken by seeking every animation to the
frame's time, so the video is exact and smooth however slow the screenshots are. Frames go straight into ffmpeg."""
import argparse
import base64
import json
import re
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from reviewer.paper import load_paper  # noqa: E402

FPS = 30
WIDTH, HEIGHT = 1920, 1080


def image_data(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


def tool_calls(trace: list[dict], agent: str) -> list[dict]:
    return [event for event in trace if event["kind"] == "tool" and event["agent"] == agent]


def example_fact(result: dict, trace: list[dict]) -> dict:
    """The fact whose validation shows the whole chain best: a conclusive verdict backed by a web page."""
    def score(check):
        has_web = any(source["kind"] == "web" and source["quotes"] for source in check["sources"])
        return (check["verdict"] in ("supported", "contradicted"), has_web, len(check["evidence"]))
    check = max(result["fact_checks"], key=score)
    calls = tool_calls(trace, f"validator:{check['id']}")
    return {"check": compact_check(check), "calls": [compact_call(call) for call in calls][:6]}


def example_finding(result: dict, trace: list[dict]) -> dict:
    """A finding whose validator ran Python, when there is one."""
    def score(check):
        ran_python = any(call["tool"] == "run_python" for call in tool_calls(trace, f"validator:{check['id']}"))
        return (ran_python, check["verdict"] != "unverifiable", len(check["evidence"]))
    check = max(result["finding_checks"], key=score)
    calls = tool_calls(trace, f"validator:{check['id']}")
    python = next((call for call in calls if call["tool"] == "run_python"), None)
    return {"check": compact_check(check), "calls": [compact_call(call) for call in calls][:7],
            "python": compact_call(python) if python else None}


def compact_check(check: dict) -> dict:
    return {"id": check["id"], "text": check.get("statement") or check.get("claim"), "verdict": check["verdict"],
            "explanation": check["explanation"], "evidence": check["evidence"][:2],
            "sources": [{"id": source["id"], "title": source["title"], "url": source["url"], "quotes": source["quotes"][:3]}
                        for source in check["sources"]][:6]}


def tool_text(output: str) -> str:
    """A tool's text; traces written before the tracer stored plain text hold the message's repr instead."""
    found = re.match(r"content=(['\"])(.*?)(\1 name=|$)", output, re.DOTALL)
    return found.group(2).replace("\\n", "\n").replace("\\'", "'") if found else output


def compact_call(call: dict) -> dict:
    return {"tool": call["tool"], "input": call["input"][:600], "output": tool_text(call["output"])[:900]}


TAG_LABELS = {"balance": "too thin", "missing": "something missing", "coherence": "loose link", "flow": "breaks the flow",
              "structure": "out of place"}


def outline(sections: list, issues: list[dict]) -> tuple[list[dict], list[dict]]:
    """The paper's sections tagged with the structure agent's issues, and the pairs of sections that repeat each other."""
    titles = [section.title for section in sections]
    rows = [{"title": section.title, "level": int(bool(re.match(r"\d+\.\d", section.title))), "size": len(section.text),
             "tags": []} for section in sections]
    repeats = {}
    for issue in issues:
        places = sorted(titles.index(name) for name in issue["section"].split(" / ") if name in titles)
        if issue["category"] == "redundancy":
            if len(places) == 2:
                pair = repeats.setdefault(tuple(places), {"from": places[0], "to": places[1], "similarity": 0})
                pair["similarity"] = max(pair["similarity"], issue.get("similarity", 0))
            continue
        label = TAG_LABELS.get(issue["category"], issue["category"])
        for place in places:
            if label not in rows[place]["tags"]:
                rows[place]["tags"].append(label)
    return rows, list(repeats.values())


def word_change(quote: str, fix: str) -> list[dict] | None:
    """The quote split around its one change: [before, {"old", "new"}, after]. None when the fix changes more than one place.

    Only what really changes is marked: "off)" → "of)" strikes "off", and "input" → "input," just inserts the comma."""
    old, new = quote.split(), fix.strip().split()
    steps = [step for step in SequenceMatcher(None, old, new).get_opcodes() if step[0] != "equal"]
    if len(steps) != 1:
        return None
    _, a, b, c, d = steps[0]
    before, after = " ".join(old[:a]), " ".join(old[b:])
    removed, added = " ".join(old[a:b]), " ".join(new[c:d])
    kept_start, kept_end = "", ""
    if removed and added.startswith(removed) or added and removed.startswith(added):
        kept_start = removed if added.startswith(removed) else added
        removed, added = removed[len(kept_start):], added[len(kept_start):]
    while removed and added and removed[-1] == added[-1] and not removed[-1].isalnum():
        kept_end, removed, added = removed[-1] + kept_end, removed[:-1], added[:-1]
    return [{"text": (before + " " if before else "") + kept_start}, {"old": removed, "new": added},
            {"text": kept_end + (" " + after if after else "")}]


def corrections(issues: list[dict], limit: int = 5) -> list[dict]:
    """Short corrections that change one place, one of each kind first."""
    usable = [{"category": issue["category"], "segments": word_change(issue["quote"], issue["suggestion"])}
              for issue in issues if len(issue["quote"]) <= 70]
    usable = [item for item in usable if item["segments"]]
    chosen = []
    for category in ("spelling", "grammar", "vocabulary", "punctuation"):
        chosen += [item for item in usable if item["category"] == category][:1]
    chosen += [item for item in usable if item not in chosen]
    return chosen[:limit]


def collect_data(run_dir: Path) -> dict:
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    trace = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    evaluation = ROOT / "evaluation"
    read = lambda name: json.loads((evaluation / name).read_text(encoding="utf-8")) if (evaluation / name).exists() else None
    llm_calls = [event for event in trace if event["kind"] == "llm"]
    sections = load_paper(ROOT / "papers" / run["paper"]).sections
    rows, repeats = outline(sections, result["structure"]["issues"])
    return {
        "paper": result["paper_title"],
        "model": run["model"],
        "minutes": run["minutes"],
        "sections": [section.title for section in sections],
        "literature": {key: result["literature"][key] for key in ("queries", "prisma", "screening_agreement")},
        "facts": [{"id": fact["id"], "statement": fact["statement"], "category": fact["category"]} for fact in result["facts"]],
        "fact_verdicts": [{"id": check["id"], "verdict": check["verdict"]} for check in result["fact_checks"]],
        "finding_verdicts": [{"id": check["id"], "verdict": check["verdict"], "text": check["claim"]} for check in result["finding_checks"]],
        "fact_example": example_fact(result, trace),
        "finding_example": example_finding(result, trace),
        "outline": rows,
        "repeats": repeats,
        "corrections": corrections(result["language"]["issues"]),
        "language_count": len(result["language"]["issues"]),
        "language_reported": result["language"]["reported"],
        "critic": result["critic"],
        "summary": result["summary"],
        "stats": {
            "model_calls": len(llm_calls),
            "tool_calls": sum(event["kind"] == "tool" for event in trace),
            "tokens": sum(event["prompt_tokens"] + event["completion_tokens"] for event in llm_calls),
            "sources": sum(len(check["sources"]) for check in result["fact_checks"] + result["finding_checks"]),
        },
        "planted": {path.stem.removeprefix("planted_"): json.loads(path.read_text(encoding="utf-8"))["rows"]
                    for path in sorted(evaluation.glob("planted_*.json"))},
        "agreement": read(f"agreement_{run['model']}.json"),
        "time": read("time.json"),
        "figures": {name: image_data(run_dir / "figures" / f"{name}.png") for name in ("verdicts", "timeline", "prisma")}
                   | {"planted": image_data(evaluation / "planted.png"), "time": image_data(evaluation / "time.png")},
    }


def build_page(run_dir: Path) -> Path:
    data = collect_data(run_dir)
    template = (HERE / "template.html").read_text(encoding="utf-8")
    page = template.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
    path = HERE / "slides.html"
    path.write_text(page, encoding="utf-8")
    return path


def record(page_path: Path, output: Path):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.goto(page_path.as_uri())
        page.wait_for_function("window.ready === true")
        seconds = page.evaluate("window.totalSeconds")
        frames = int(seconds * FPS)
        print(f"recording {seconds:.0f} s, {frames} frames")
        encoder = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(output)],
            stdin=subprocess.PIPE)
        for frame in range(frames):
            page.evaluate(f"window.seek({frame / FPS})")
            encoder.stdin.write(page.screenshot(type="jpeg", quality=95))
            if frame % (FPS * 10) == 0:
                print(f"  {frame / FPS:.0f} s", flush=True)
        encoder.stdin.close()
        encoder.wait()
        browser.close()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="the run folder to present")
    arguments = parser.parse_args()
    page = build_page(Path(arguments.run))
    output = HERE / "paper_review_pipeline.mp4"
    record(page, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
