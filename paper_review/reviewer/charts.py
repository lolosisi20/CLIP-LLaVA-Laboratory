import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
VERDICT_COLORS = {"supported": "#0ca30c", "partially_supported": "#fab219", "contradicted": "#d03b3b", "unverifiable": MUTED}
VERDICT_LABELS = {"supported": "Supported", "partially_supported": "Partly supported", "contradicted": "Contradicted", "unverifiable": "Unverifiable"}
LABEL_ON = {"supported": "#ffffff", "partially_supported": INK, "contradicted": "#ffffff", "unverifiable": "#ffffff"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_SECONDARY, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": INK_SECONDARY, "font.family": "Segoe UI", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.titlelocation": "left", "axes.grid": False,
})


def save(figure, path: Path):
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


# ── PRISMA flow ──

def box(axes, x, y, width, height, text):
    axes.add_patch(Rectangle((x, y), width, height, facecolor="white", edgecolor=INK_SECONDARY, linewidth=1))
    axes.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9.5, color=INK, wrap=True)


def arrow(axes, start, end):
    axes.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.4})


def prisma_chart(prisma: dict, path: Path):
    identified = prisma["identified"]
    removed = prisma["removed_before_screening"]
    figure, axes = plt.subplots(figsize=(9, 6.2))
    axes.set_xlim(0, 10)
    axes.set_ylim(0, 10)
    axes.axis("off")
    axes.set_title("Prior work: PRISMA 2020 flow", pad=4)

    sources = "\n".join(f"{name.replace('_', ' ')}: {count}" for name, count in identified.items())
    box(axes, 0.3, 7.6, 4.4, 2.0, f"Records identified\n{sum(identified.values())}\n{sources}")
    box(axes, 5.6, 7.6, 4.1, 2.0, f"Removed before screening\nduplicates: {prisma['duplicates_removed']}\n"
        + "\n".join(f"{reason}: {count}" for reason, count in removed.items()))
    box(axes, 0.3, 4.6, 4.4, 1.6, f"Records screened\n(title and abstract, two screeners)\n{prisma['screened']}")
    box(axes, 5.6, 4.6, 4.1, 1.6, f"Records excluded\n{prisma['excluded']}")
    box(axes, 0.3, 1.4, 4.4, 1.8, f"Included as evidence\nfor the validators\n{prisma['included']}")
    arrow(axes, (2.5, 7.6), (2.5, 6.2))
    arrow(axes, (4.7, 8.6), (5.6, 8.6))
    arrow(axes, (2.5, 4.6), (2.5, 3.2))
    arrow(axes, (4.7, 5.4), (5.6, 5.4))
    save(figure, path)


# ── Verdicts ──

def verdict_chart(fact_checks: list[dict], finding_checks: list[dict], path: Path):
    groups = {"Stated facts": fact_checks, "Findings": finding_checks}
    figure, axes = plt.subplots(figsize=(9, 2.8))
    for row, (name, checks) in enumerate(groups.items()):
        counts = Counter(check["verdict"] for check in checks)
        left = 0
        for verdict, color in VERDICT_COLORS.items():
            count = counts.get(verdict, 0)
            if not count:
                continue
            axes.barh(row, count, left=left, color=color, height=0.55, edgecolor=SURFACE, linewidth=2)
            axes.text(left + count / 2, row, str(count), ha="center", va="center", color=LABEL_ON[verdict], fontweight="bold")
            left += count
    axes.set_yticks(range(len(groups)), list(groups))
    axes.invert_yaxis()
    axes.set_xlabel("claims")
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.spines["left"].set_visible(False)
    axes.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in VERDICT_COLORS.values()]
    axes.legend(handles, VERDICT_LABELS.values(), ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    axes.set_title("Validation verdicts", pad=28)
    save(figure, path)


# ── Issues by section ──

def issues_chart(structure: dict, language: dict, path: Path):
    by_section = defaultdict(lambda: Counter())
    for issue in structure["issues"]:
        by_section[issue["section"].split(" / ")[0]]["Structure and redundancy"] += 1
    for issue in language["issues"]:
        by_section[issue["section"]]["Grammar and vocabulary"] += 1
    sections = sorted(by_section, key=lambda section: -sum(by_section[section].values()))[:14]
    kinds = ["Structure and redundancy", "Grammar and vocabulary"]

    figure, axes = plt.subplots(figsize=(9, 0.42 * len(sections) + 1.4))
    for position, section in enumerate(sections):
        left = 0
        for kind, color in zip(kinds, SERIES):
            count = by_section[section][kind]
            if count:
                axes.barh(position, count, left=left, color=color, height=0.6, edgecolor=SURFACE, linewidth=2)
                left += count
        axes.text(left + 0.1, position, str(left), va="center", color=INK_SECONDARY, fontsize=9)
    axes.set_yticks(range(len(sections)), [section[:48] for section in sections])
    axes.invert_yaxis()
    axes.set_xlabel("issues")
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.spines["left"].set_visible(False)
    axes.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in SERIES[:2]]
    axes.legend(handles, kinds, ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    axes.set_title("Writing issues by section", pad=28)
    save(figure, path)


# ── Agent timeline and cost, from the trace ──

AGENT_FAMILIES = ["literature", "facts", "findings", "structure", "language", "fact validators", "finding validators", "critic"]


def family_of(agent: str, finding_ids: set[str]) -> str:
    if agent.startswith("validator:"):
        return "finding validators" if agent.split(":")[1] in finding_ids else "fact validators"
    if agent in ("search_strategy", "screener_a", "screener_b", "screening_adjudicator", "retrieval"):
        return "literature"
    if agent == "redundancy":
        return "structure"
    if agent in ("critic", "synthesis"):
        return "critic"
    return agent if agent in AGENT_FAMILIES else "critic"


def read_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def timeline_chart(events: list[dict], finding_ids: set[str], path: Path):
    spans = [event for event in events if event["kind"] in ("llm", "tool")]
    figure, axes = plt.subplots(figsize=(10, 4.2))
    for event in spans:
        family = family_of(event["agent"], finding_ids)
        row = AGENT_FAMILIES.index(family)
        is_model_call = event["kind"] == "llm"
        axes.barh(row, event["seconds"] / 60, left=event["start"] / 60, height=0.6 if is_model_call else 0.25,
                  color="#9ec5f4" if is_model_call else SERIES[0], linewidth=0)
    axes.set_yticks(range(len(AGENT_FAMILIES)), AGENT_FAMILIES)
    axes.invert_yaxis()
    axes.set_xlabel("minutes since the start of the run")
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.spines["left"].set_visible(False)
    axes.tick_params(axis="y", length=0)
    axes.set_title("When each agent worked (wide bars: model calls, thin bars: tool calls)", pad=8)
    save(figure, path)


def cost_chart(events: list[dict], finding_ids: set[str], path: Path):
    tokens = Counter()
    for event in events:
        if event["kind"] == "llm":
            tokens[family_of(event["agent"], finding_ids)] += event["prompt_tokens"] + event["completion_tokens"]
    families = [family for family in AGENT_FAMILIES if tokens[family]]
    figure, axes = plt.subplots(figsize=(9, 3.6))
    values = [tokens[family] / 1000 for family in families]
    axes.barh(range(len(families)), values, color=SERIES[0], height=0.6)
    for position, value in enumerate(values):
        axes.text(value, position, f" {value:,.0f}k", va="center", color=INK_SECONDARY, fontsize=9)
    axes.set_yticks(range(len(families)), families)
    axes.invert_yaxis()
    axes.set_xlabel("thousand tokens (prompt + completion)")
    axes.spines["left"].set_visible(False)
    axes.tick_params(axis="y", length=0)
    axes.set_title("Tokens used by each part of the pipeline", pad=8)
    save(figure, path)


def draw_all(result: dict, run_dir: Path) -> list[Path]:
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    events = read_trace(run_dir / "trace.jsonl")
    finding_ids = {check["id"] for check in result["finding_checks"]}
    charts = {
        "prisma.png": lambda path: prisma_chart(result["literature"]["prisma"], path),
        "verdicts.png": lambda path: verdict_chart(result["fact_checks"], result["finding_checks"], path),
        "issues_by_section.png": lambda path: issues_chart(result["structure"], result["language"], path),
        "timeline.png": lambda path: timeline_chart(events, finding_ids, path),
        "tokens.png": lambda path: cost_chart(events, finding_ids, path),
    }
    for name, draw in charts.items():
        draw(figures / name)
    return [figures / name for name in charts]
