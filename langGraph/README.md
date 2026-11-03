# Intelligent University Assistant

A small agent built with LangGraph. It reads a student's question, decides what kind of question it is, and sends it to a specialized agent. The agent can use tools (calculator, date and time, course database) and remembers the conversation.

## Setup
```bash
pip install -U langgraph langchain langchain-openai
```

## Run
```bash
cd source
python university_assistant.py                                  # chat in the console
python university_assistant.py "Who teaches Deep Learning?"     # ask one question
```
Or open `langgraph_lab.ipynb` and run the cells from top to bottom.

Environment variables: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`. Without them it uses my own model, Swift 1.5 27B (a Qwen 27B fine-tune), through its OpenAI-compatible API. To use OpenAI instead, set `LLM_BASE_URL=https://api.openai.com/v1`, your key, and for example `LLM_MODEL=gpt-4o-mini`. In that case also remove `reasoning_effort="low"`, since not every model accepts it.

## Design choices
- **One classifier, five agents.** The classifier only picks the category, and each agent has its own system prompt.
- **Tools go back to the same agent.** After a tool runs, the agent that asked for it writes the final answer.
- **Safe calculator.** It parses the expression with `ast` instead of using `eval()`, so it can only do arithmetic.
- **Errors don't crash the graph.** Empty questions and API errors go to a fallback answer, unknown categories become GENERAL, and bad tool calls come back as messages the agent can recover from.
- **Memory per conversation.** `MemorySaver` with a `thread_id` keeps each conversation separate. It lives in memory, so it's lost when the program stops.
- **Course data is made up.** The SQLite database holds sample courses just to show the agent using a tool.

## Files
`langgraph_lab.ipynb` (executed notebook), `source/university_assistant.py`, `architecture.png`, `screenshots/`, `report.pdf`.
