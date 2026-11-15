# Agentic Paper Review

I built a set of AI agents that review a scientific paper. They pull out the facts the paper states and check each
one on the web, test the paper's results against earlier work, look at how the paper is organized, and find grammar
mistakes. Each verdict has to quote a source the agent read.

I tested it on the FlashAttention paper (Dao et al., 2022).

## How it works

The pipeline is one LangGraph graph (`reviewer/graph.py`). After the paper is read, five agents start together:

- the facts agent lists what the paper takes as true about the world, such as hardware specs
- the findings agent lists what the authors claim about their own work, such as speedups
- the literature agent searches arXiv, OpenAlex and Semantic Scholar, and two screeners pick the relevant papers
  (PRISMA)
- the structure agent checks the outline and finds sections that repeat each other
- the language agent finds grammar and spelling errors

Then a new agent checks each fact and each finding on its own. It can search the web, read pages, search the earlier
papers and run Python. With `--human`, I can look over the facts before they are checked. Every quote an agent gives
is matched against the page it came from, and a verdict without a real quote becomes "unverifiable". A critic goes
over all the verdicts before the report is written.

Search over the paper and its sources mixes keyword search (BM25) with embeddings (EmbeddingGemma). Every model call
and tool call is logged in `trace.jsonl`.

## Results

One run with Swift 1.5 27B on a 16 GB GPU took 28 minutes. The report is in
`runs/flashattention_swift-1-5-27b-xxs_20261115-190315/report.md`.

| | |
|---|---|
| Earlier work | 17 papers found, 6 kept after screening |
| Facts | 20 checked: 10 supported, 3 partly supported, 7 unverifiable |
| Findings | 8 checked, all supported |
| Structure | 9 issues |
| Language | 17 errors |

![Verdicts](runs/flashattention_swift-1-5-27b-xxs_20261115-190315/figures/verdicts.png)

To evaluate it, I hid 10 mistakes in a copy of the paper, and it caught 8. When new agents checked the same claims a
second time, 79% of the verdicts matched. A review by hand would take about 7.4 hours, so the pipeline is about 16
times faster. The numbers are in `evaluation/`.

## Setup

```bash
pip install -r requirements.txt
playwright install chromium   # only for the video
```

Copy `.env.example` to `.env` and set the address of your model server. It needs an OpenAI-compatible chat endpoint
with tool calling, plus web search and embeddings. The review settings, such as the sources and how many facts to
check, are in `protocol.toml`.

## Usage

```bash
python review.py papers/flashattention.pdf            # review a paper
python review.py papers/flashattention.pdf --human    # look over the facts before they are checked
python evaluate.py planted papers/flashattention.pdf  # hidden-mistakes test
python evaluate.py agreement runs/<run>               # check the same claims again
python evaluate.py time runs/<run>                    # compare with a review by hand
python video/build_video.py runs/<run>                # build the video
```

Each run writes a folder in `runs/` with the report, the full results, the trace and the charts.

## Limits

- The agents check claims against sources. They don't rerun the paper's experiments.
- `run_python` runs code the model wrote, so use it on a machine where that's acceptable.
- The model can misread a source. The report shows every quote, so a reader can check it.

## Team

I did this project alone.

| Member | Contributions |
|---|---|
| Saloua Oulad Sine | Everything: the pipeline and its agents, retrieval, the literature review, the validators, the evaluation, the report and the video |
