import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import settings
from .trace import log_step, traced

class SlotLimitedTransport(httpx.HTTPTransport):
    """As many requests reach the model at once as it has slots, however many agents run; the rest wait their turn."""
    slots: threading.Semaphore | None = None
    creating = threading.Lock()

    def handle_request(self, request):
        with self.creating:
            if SlotLimitedTransport.slots is None:
                SlotLimitedTransport.slots = threading.Semaphore(settings.profile["slots"])
        with self.slots:
            response = super().handle_request(request)
            response.read()
            return response


shared_client = httpx.Client(transport=SlotLimitedTransport(), timeout=900)


def chat_model(max_tokens: int = 3000, is_agent: bool = False) -> ChatOpenAI:
    """Agents that work with tools think first; one-shot answers do not."""
    effort = settings.profile["agent_effort" if is_agent else "answer_effort"]
    return ChatOpenAI(
        base_url=settings.llm_url,
        api_key=settings.api_key,
        model=settings.model,
        max_tokens=max_tokens,
        timeout=900,
        max_retries=2,
        http_client=shared_client,
        **settings.profile["sampling"],
        **({"reasoning_effort": effort} if effort else {}),
    )


def messages_for(prompt: str, system: str | None) -> list:
    messages = [SystemMessage(content=system)] if system else []
    return messages + [HumanMessage(content=prompt)]


def ask(prompt: str, agent: str, system: str | None = None, max_tokens: int = 3000) -> str:
    reply = chat_model(max_tokens).invoke(messages_for(prompt, system), config=traced(agent))
    return reply.content.strip()


def ask_json[T: BaseModel](prompt: str, schema: type[T], agent: str, system: str | None = None,
                           max_tokens: int = 4000, attempts: int = 3, think: bool = False) -> T | None:
    """One structured answer, decoded against the schema by the server's grammar. None when every attempt fails."""
    model = chat_model(max_tokens, is_agent=think).with_structured_output(schema, method="json_schema")
    for attempt in range(1, attempts + 1):
        try:
            return model.invoke(messages_for(prompt, system), config=traced(agent))
        except Exception as error:
            log_step(agent, f"structured answer failed (attempt {attempt}): {type(error).__name__}: {str(error)[:200]}")
    return None


def parallel_map(function, items: list, workers: int) -> list:
    """Runs one agent per item, `workers` at a time, keeping the order of the items."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(function, items))
