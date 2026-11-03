import ast
import operator as op
import os
import sqlite3
import sys
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

llm = ChatOpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "https://orchestrator.dvnkboi.com/api/llm/v1"),
    api_key=os.environ.get("LLM_API_KEY", "not-needed"),
    model=os.environ.get("LLM_MODEL", "swift-1.5-27b-xxs"),
    temperature=0,
    reasoning_effort="low",
)

OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
    ast.USub: op.neg, ast.UAdd: op.pos,
}


def safe_eval(node):
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression such as '25 * 4 + 10'. Supports + - * / ** % //."""
    try:
        return str(safe_eval(ast.parse(expression.strip(), mode="eval")))
    except Exception:
        return "Invalid expression"


@tool
def current_datetime(_: str = "") -> str:
    """Return the current date and time. Use for any question about today's date or the current time."""
    return datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")


db = sqlite3.connect(":memory:", check_same_thread=False)
db.execute("create table courses (name text, teacher text, semester int, exam text)")
db.executemany("insert into courses values (?, ?, ?, ?)", [
    ("Machine Learning", "Dr. Alaoui", 1, "15 January"),
    ("Deep Learning", "Dr. Bennani", 2, "10 June"),
    ("Generative AI", "Dr. Idrissi", 2, "17 June"),
    ("Databases", "Dr. Tazi", 1, "20 January"),
])


@tool
def course_info(course: str) -> str:
    """Look up a course in the university database. Returns its teacher, semester and exam date."""
    rows = db.execute("select * from courses where name like ?", (f"%{course}%",)).fetchall()
    if not rows:
        return f"No course found for '{course}'."
    return "\n".join(f"{n}: taught by {t}, semester {s}, exam on {e}" for n, t, s, e in rows)


tools = [calculator, current_datetime, course_info]
llm_with_tools = llm.bind_tools(tools)

CATEGORIES = ["COURSES", "PROGRAMMING", "MATHEMATICS", "ADMINISTRATION", "GENERAL"]
NODE_NAMES = {c: f"{c.lower()}_agent" for c in CATEGORIES}

SYSTEM_PROMPTS = {
    "COURSES": "You are a university courses advisor. Explain course content, prerequisites, schedules and study "
               "strategies. Use the course_info tool for questions about a specific course. If you lack "
               "institution-specific data, say so honestly.",
    "PROGRAMMING": "You are a programming tutor. Give clear explanations and correct Python code when appropriate.",
    "MATHEMATICS": "You are a mathematics tutor. Show steps. ALWAYS use the calculator tool for arithmetic.",
    "ADMINISTRATION": "You are a university administration assistant (enrollment, fees, certificates, procedures). "
                      "Never invent official rules or deadlines; recommend contacting the administration office "
                      "when unsure.",
    "GENERAL": "You are a helpful university assistant. Use the date/time tool for questions about the current "
               "date or time.",
}


class AssistantState(TypedDict):
    messages: Annotated[list, add_messages]
    category: str


def classify(state: AssistantState):
    text = state["messages"][-1].content.strip()
    if not text:
        return {"category": "EMPTY"}

    prompt = f"""
    Classify the student's question into exactly one category:
    COURSES (course content, syllabus, schedules, exams, grading),
    PROGRAMMING, MATHEMATICS,
    ADMINISTRATION (enrollment, fees, certificates, procedures, deadlines),
    GENERAL (anything else).

    Question: {text}

    Return only the category name.
    """
    try:
        raw = llm.invoke(prompt).content.strip().upper()
    except Exception as e:
        print("Classifier API error:", e)
        return {"category": "ERROR"}
    return {"category": next((c for c in CATEGORIES if c in raw), "GENERAL")}


def make_agent(category: str):
    system = SystemMessage(content=SYSTEM_PROMPTS[category])

    def node(state: AssistantState):
        try:
            return {"messages": [llm_with_tools.invoke([system] + state["messages"])]}
        except Exception as e:
            return {"messages": [AIMessage(
                content=f"Sorry, I ran into a technical problem while answering ({type(e).__name__}). Please try again."
            )]}
    return node


def fallback(state: AssistantState):
    if state["category"] == "EMPTY":
        msg = "Your question seems to be empty. Please type a question and I'll help."
    else:
        msg = "Sorry, I couldn't process your question right now. Please try again in a moment."
    return {"messages": [AIMessage(content=msg)]}


def route_after_classifier(state: AssistantState):
    return NODE_NAMES.get(state["category"], "fallback")


def route_after_agent(state: AssistantState):
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END


def route_after_tools(state: AssistantState):
    return NODE_NAMES.get(state["category"], "general_agent")


def build_assistant():
    builder = StateGraph(AssistantState)
    builder.add_node("classifier", classify)
    builder.add_node("fallback", fallback)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    for c in CATEGORIES:
        builder.add_node(NODE_NAMES[c], make_agent(c))

    builder.add_edge(START, "classifier")
    builder.add_conditional_edges(
        "classifier", route_after_classifier,
        {**{n: n for n in NODE_NAMES.values()}, "fallback": "fallback"},
    )
    for c in CATEGORIES:
        builder.add_conditional_edges(NODE_NAMES[c], route_after_agent, {"tools": "tools", END: END})
    builder.add_conditional_edges("tools", route_after_tools, {n: n for n in NODE_NAMES.values()})
    builder.add_edge("fallback", END)
    return builder.compile(checkpointer=MemorySaver())


def ask(assistant, question, thread_id="cli"):
    config = {"configurable": {"thread_id": thread_id}}
    result = assistant.invoke({"messages": [HumanMessage(content=question)]}, config)
    return result["category"], result["messages"][-1].content


if __name__ == "__main__":
    assistant = build_assistant()

    if len(sys.argv) > 1:
        category, answer = ask(assistant, " ".join(sys.argv[1:]))
        print(f"[{category}] {answer}")
        sys.exit()

    print("University assistant. Type quit to leave.")
    while True:
        question = input("\nYou: ")
        if question.strip().lower() in ("quit", "exit"):
            break
        category, answer = ask(assistant, question)
        print(f"[{category}] {answer}")
