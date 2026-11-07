"""LangGraph coding-agent boilerplate using Google AI Studio / Gemini.

Install:
    pip install -U langgraph langchain-core langchain-google-genai

Set your free Google AI Studio key before running:
    $env:GOOGLE_API_KEY = "..."
    python agents.py "Inspect this repository and explain its structure"

The agent has two workflows:
    react - tool-driven ReAct loop (reason/tool/observe until finished)
    plan - create a short plan, then execute it with tools

This is intentionally a starter: add project-specific tools and approval gates
before allowing it to modify production repositories.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.parse
import urllib.request
import re
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_FILE_BYTES = 30_000
MAX_AGENT_STEPS = 20
MAX_CHAT_MESSAGES = 12
load_dotenv(PROJECT_ROOT / ".env")


def _safe_path(relative_path: str) -> Path:
    """Resolve a project-relative path and reject paths outside PROJECT_ROOT."""
    candidate = (PROJECT_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must stay inside {PROJECT_ROOT}: {relative_path}") from exc
    return candidate


def _confirm_change(action: str, path: str) -> bool:
    """Ask the human before allowing a tool to modify a local file."""
    if os.getenv("AUTO_APPROVE_WRITES", "false").lower() in {"1", "true", "yes"}:
        return True
    answer = input(f"\n[approval] Agent wants to {action} '{path}'. Allow? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


@tool
def list_files(path: str = ".") -> str:
    """List files below a project-relative directory, excluding common generated folders."""
    root = _safe_path(path)
    if not root.is_dir():
        return f"Not a directory: {path}"
    ignored = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
    files = [
        str(item.relative_to(PROJECT_ROOT))
        for item in root.rglob("*")
        if item.is_file() and not ignored.intersection(item.parts)
    ]
    return "\n".join(sorted(files)[:200]) or "No files found."


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file inside the project, capped to prevent huge context dumps."""
    file_path = _safe_path(path)
    if not file_path.is_file():
        return f"File not found: {path}"
    if file_path.stat().st_size > MAX_FILE_BYTES:
        return f"Refusing to read {path}: file exceeds {MAX_FILE_BYTES} bytes."
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Refusing to read {path}: it is not UTF-8 text."


@tool
def search_files(query: str, path: str = ".") -> str:
    """Search UTF-8 project files for a literal string and return matching lines."""
    root = _safe_path(path)
    if not root.is_dir():
        return f"Not a directory: {path}"
    results: list[str] = []
    ignored = {".git", ".venv", "venv", "node_modules", "__pycache__"}
    for file_path in root.rglob("*"):
        if not file_path.is_file() or ignored.intersection(file_path.parts):
            continue
        try:
            if file_path.stat().st_size > MAX_FILE_BYTES:
                continue
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(lines, start=1):
            if query.lower() in line.lower():
                results.append(f"{file_path.relative_to(PROJECT_ROOT)}:{number}: {line}")
                if len(results) >= 40:
                    return "\n".join(results)
    return "\n".join(results) or "No matches."


@tool
def write_file(path: str, content: str) -> str:
    """Write a UTF-8 file inside the project. The model should ask for approval first."""
    file_path = _safe_path(path)
    if not _confirm_change("write", path):
        return "Write cancelled by the user."
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Wrote {file_path.relative_to(PROJECT_ROOT)} ({len(content)} characters)."


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace one exact occurrence in a UTF-8 project file."""
    file_path = _safe_path(path)
    content = file_path.read_text(encoding="utf-8")
    occurrences = content.count(old_text)
    if occurrences != 1:
        return f"Edit aborted: expected exactly one occurrence, found {occurrences}."
    if not _confirm_change("edit", path):
        return "Edit cancelled by the user."
    file_path.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {file_path.relative_to(PROJECT_ROOT)}."


@tool
def git_diff() -> str:
    """Return the current git diff for the project."""
    result = subprocess.run(
        ["git", "diff", "--", "."], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout or result.stderr or "No diff."


@tool
def github_search(query: str, language: str = "") -> str:
    """Search public GitHub repositories/code metadata without requiring a GitHub token."""
    search_query = query + (f" language:{language}" if language else "")
    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
        {"q": search_query, "sort": "stars", "per_page": 5}
    )
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "langgraph-coding-agent"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except Exception as exc:  # network/API failures should be visible to the agent
        return f"GitHub search failed: {exc}"
    return "\n".join(
        f"- {item['full_name']}: {item.get('description') or 'No description'} ({item['html_url']})"
        for item in payload.get("items", [])
    ) or "No repositories found."


TOOLS = [list_files, read_file, search_files, write_file, edit_file, git_diff, github_search]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: str
    mode: Literal["react", "plan"]


SYSTEM_PROMPT = """You are a careful AI coding agent working in the project directory.
Use tools to inspect facts before making claims. Keep changes focused. Before using
write_file or edit_file, explain the intended change in your response and verify the
target with read_file. Never access files outside the project root. Use git_diff to
review changes after editing. Do not expose hidden chain-of-thought; provide concise
reasoning summaries, tool results, and decisions instead."""


def _model() -> ChatGoogleGenerativeAI:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Copy .env.example to .env, put your key "
            "after GOOGLE_API_KEY=, save it, and run the program again."
        )
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        temperature=0,
        max_retries=1,
        max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048")),
    )


def build_graph(mode: Literal["react", "plan"] = "react"):
    model = _model().bind_tools(TOOLS)
    tool_node = ToolNode(TOOLS)

    def react_agent(state: AgentState) -> AgentState:
        response = model.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    def make_plan(state: AgentState) -> AgentState:
        planner = _model()
        prompt = ("Create a concise numbered implementation plan for the user's request. "
                  "Inspect the repository with tools only after planning; do not edit yet.\n\n" +
                  str(state["messages"][-1].content))
        response = planner.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
        return {"plan": str(response.content), "messages": [AIMessage(content=f"Plan:\n{response.content}")]}

    def execute(state: AgentState) -> AgentState:
        response = model.invoke([
            SystemMessage(content=SYSTEM_PROMPT + "\nFollow the plan already in the conversation. "
                          "Use tools as needed, then summarize changes and verification."),
            *state["messages"],
        ])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    if mode == "react":
        graph.add_node("agent", react_agent)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
    else:
        graph.add_node("plan", make_plan)
        graph.add_node("execute", execute)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "execute")
        graph.add_conditional_edges("execute", tools_condition, {"tools": "tools", END: END})
        graph.add_edge("tools", "execute")
    return graph.compile()


def _status_for_update(node_name: str, update: AgentState) -> str:
    """Return a safe action summary; do not print hidden chain-of-thought."""
    if node_name == "plan":
        return "Creating an implementation plan..."
    if node_name == "tools":
        messages = update.get("messages", [])
        count = sum(isinstance(message, ToolMessage) for message in messages)
        return f"Finished using {count} tool{'s' if count != 1 else ''}."
    if node_name in {"agent", "execute"}:
        for message in update.get("messages", []):
            if isinstance(message, AIMessage) and message.tool_calls:
                names = ", ".join(call["name"] for call in message.tool_calls)
                return f"Using tools: {names}..."
        return "Generating a response..."
    return f"Working on {node_name}..."


def _run_turn(
    request: str,
    mode: Literal["react", "plan"],
    history: list[BaseMessage] | None = None,
    show_updates: bool = False,
) -> tuple[str, list[BaseMessage]]:
    """Run one turn and return the response plus updated conversation history."""
    graph = build_graph(mode)
    messages = [*(history or []), HumanMessage(content=request)]
    final_state: AgentState = {"mode": mode, "messages": messages}
    if show_updates:
        print(f"[status] Starting {mode} mode...")
    for update in graph.stream(
        final_state,
        stream_mode="updates",
        config={"recursion_limit": MAX_AGENT_STEPS},
    ):
        for node_name, node_update in update.items():
            final_state["messages"] = [
                *final_state.get("messages", []),
                *node_update.get("messages", []),
            ]
            if "plan" in node_update:
                final_state["plan"] = node_update["plan"]
            if show_updates:
                print(f"[status] {_status_for_update(node_name, node_update)}")
    response = str(final_state["messages"][-1].content)
    # Retain dialogue, not bulky tool calls/results, across chat turns.
    next_history = [*(history or []), HumanMessage(content=request), AIMessage(content=response)]
    return response, next_history[-MAX_CHAT_MESSAGES:]


def run_agent(
    request: str,
    mode: Literal["react", "plan"] = "react",
    show_updates: bool = False,
) -> str:
    """Run one standalone request."""
    response, _ = _run_turn(request, mode, show_updates=show_updates)
    return response


def chat_loop(mode: Literal["react", "plan"] = "react") -> None:
    """Run a multi-turn chatbot session with agent tools and conversation memory."""
    history: list[BaseMessage] = []
    print(f"AI coding agent ready (model: {os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite')}). "
          "Type /help for commands, or /exit to quit.")
    while True:
        try:
            request = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if not request:
            continue
        if request in {"/exit", "/quit"}:
            print("Goodbye!")
            return
        if request == "/help":
            print("Commands: /mode react, /mode plan, /model [model-name], /clear, /exit")
            continue
        if request == "/clear":
            history.clear()
            print("Conversation cleared.")
            continue
        if request.startswith("/mode "):
            new_mode = request.split(maxsplit=1)[1].strip().lower()
            if new_mode in {"react", "plan"}:
                mode = new_mode  # type: ignore[assignment]
                print(f"Switched to {mode} mode.")
            else:
                print("Mode must be 'react' or 'plan'.")
            continue
        if request == "/model":
            print(f"Current model: {os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite')}")
            continue
        if request.startswith("/model "):
            model_name = request.split(maxsplit=1)[1].strip()
            if not model_name or any(char.isspace() for char in model_name):
                print("Enter one model ID, such as /model gemini-3.8-flash")
            else:
                os.environ["GEMINI_MODEL"] = model_name
                print(f"Switched to {model_name} for this chat session.")
            continue
        try:
            response, history = _run_turn(request, mode, history, show_updates=True)
            print(f"\nAgent: {response}")
        except Exception as exc:
            error_text = str(exc)
            if "503" in error_text and "UNAVAILABLE" in error_text:
                print("\nGoogle's model is temporarily busy (503). Your chat is still open; "
                      "try your message again later or use /model with another model "
                      "available to your API key.")
            elif "404" in error_text and "NOT_FOUND" in error_text:
                print("\nThis model is unavailable to your API key (404). "
                      "Try /model gemini-3.8-flash, then send your message again.")
            else:
                print(f"\nAgent error: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph Gemini coding agent")
    parser.add_argument("request", nargs="*", help="What the coding agent should do")
    parser.add_argument("--mode", choices=("react", "plan"), default="react")
    parser.add_argument("--chat", action="store_true", help="Start an interactive multi-turn chat")
    args = parser.parse_args()
    request = " ".join(args.request).strip()
    if args.chat or not request:
        chat_loop(args.mode)
        raise SystemExit(0)
    if not request:
        parser.error("please provide a request, for example: python agents.py \"list the files\"")
    print(run_agent(request, args.mode, show_updates=True))
