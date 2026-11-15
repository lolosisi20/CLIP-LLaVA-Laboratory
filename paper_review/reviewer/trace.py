import json
import threading
import time
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler


class Tracer(BaseCallbackHandler):
    """Writes every model call and tool call of every agent to trace.jsonl, with timings and token counts."""

    def __init__(self, path: Path):
        self.path = path
        self.started = {}
        self.lock = threading.Lock()
        self.origin = time.time()

    def write(self, **event):
        event["t"] = round(time.time() - self.origin, 3)
        with self.lock, self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        self.started[run_id] = (time.time(), (metadata or {}).get("agent", "?"))

    def on_llm_end(self, response, *, run_id, **kwargs):
        began, agent = self.started.pop(run_id, (time.time(), "?"))
        usage = (response.llm_output or {}).get("token_usage") or {}
        message = response.generations[0][0].text if response.generations else ""
        self.write(kind="llm", agent=agent, start=round(began - self.origin, 3), seconds=round(time.time() - began, 2),
                   prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=usage.get("completion_tokens", 0),
                   output=message[:2000])

    def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, **kwargs):
        self.started[run_id] = (time.time(), (metadata or {}).get("agent", "?"), serialized.get("name"), input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        began, agent, name, tool_input = self.started.pop(run_id, (time.time(), "?", "?", ""))
        output = getattr(output, "content", output)
        self.write(kind="tool", agent=agent, tool=name, start=round(began - self.origin, 3),
                   seconds=round(time.time() - began, 2), input=str(tool_input)[:1000], output=str(output)[:2000])

    def on_tool_error(self, error, *, run_id, **kwargs):
        self.on_tool_end(f"ERROR: {error}", run_id=run_id)

    def step(self, agent: str, message: str, **data):
        self.write(kind="step", agent=agent, message=message, **data)


tracer: Tracer | None = None


def start_trace(path: Path) -> Tracer:
    global tracer
    tracer = Tracer(path)
    return tracer


def traced(agent: str) -> dict:
    """The run config that tags every call below it with the agent's name."""
    callbacks = [tracer] if tracer else []
    return {"callbacks": callbacks, "metadata": {"agent": agent}}


def log_step(agent: str, message: str, **data):
    print(f"[{agent}] {message}", flush=True)
    if tracer:
        tracer.step(agent, message, **data)
