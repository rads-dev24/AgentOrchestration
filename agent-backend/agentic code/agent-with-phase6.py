"""
5G QoS Orchestration Agent — Phase 6
ReAct agent using LangGraph + Ollama (local LLM with tool calling)
Enhanced for Phase 6: iperf3 traffic violation detection + correction + audit
"""

import json
import time
import datetime
from typing import Annotated, TypedDict, List

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.rule import Rule
from rich import box
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

console = Console()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
NEF_BASE    = "http://127.0.0.1:8000"
OLLAMA_BASE = "http://localhost:11434"
LLM_MODEL   = "qwen2.5:7b"

OBSERVE_INTERVAL_SECONDS = 20   # tighter for Phase 6 validation
MAX_LOOP_ITERATIONS      = 100

# Phase 6: known baseline policies per IMSI — agent uses this to detect violations
# Add all your registered IMSIs and their EXPECTED (correct) policy here
BASELINE_POLICIES = {
    "208930000000001": "default",
    "208930000000002": "default",
    # add more as needed
}


# ─────────────────────────────────────────────
# Tool definitions
# ─────────────────────────────────────────────

@tool
def get_core_status() -> str:
    """Get the health and status of the 5G core (Ella-Core). Use this first to check if the core is up."""
    try:
        r = requests.get(f"{NEF_BASE}/core/status", timeout=10)
        r.raise_for_status()
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return f"ERROR: {e}"


@tool
def get_active_sessions() -> str:
    """List all currently registered UEs with active PDU sessions. Returns IMSI, registration status, and QoS info."""
    try:
        r = requests.get(f"{NEF_BASE}/core/sessions", timeout=10)
        r.raise_for_status()
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return f"ERROR: {e}"


@tool
def get_subscriber_session(imsi: str) -> str:
    """
    Get detailed session information for a specific UE by IMSI.
    Args:
        imsi: The IMSI of the subscriber (e.g. '208930000000001')
    """
    try:
        r = requests.get(f"{NEF_BASE}/core/sessions/{imsi}", timeout=10)
        if r.status_code == 404:
            return f"Subscriber {imsi} not found."
        r.raise_for_status()
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return f"ERROR: {e}"


@tool
def get_subscriber_policy(imsi: str) -> str:
    """
    Fetch the QoS policy currently assigned to a subscriber.
    Args:
        imsi: The IMSI of the subscriber
    """
    try:
        r = requests.get(f"{NEF_BASE}/core/policy/{imsi}", timeout=10)
        if r.status_code == 404:
            return f"No policy found for {imsi}."
        r.raise_for_status()
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return f"ERROR: {e}"


@tool
def update_subscriber_ambr(imsi: str, policy_name: str) -> str:
    """
    Update the QoS policy (AMBR) for a subscriber. Use this to throttle or restore bandwidth.
    Args:
        imsi: The IMSI of the subscriber to update
        policy_name: The name of the policy to assign (e.g. 'default', 'premium', 'throttled')
    """
    try:
        r = requests.put(
            f"{NEF_BASE}/core/subscriber/{imsi}/ambr",
            json={"policy_name": policy_name},
            timeout=10
        )
        r.raise_for_status()
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return f"ERROR: {e}"


@tool
def get_metrics_snapshot() -> str:
    """Fetch a Prometheus metrics snapshot from Ella-Core. Useful for observing throughput, latency, and session counts."""
    try:
        r = requests.get(f"{NEF_BASE}/core/metrics", timeout=10)
        r.raise_for_status()
        data = r.json()
        metrics_text = data.get("metrics", "")
        if len(metrics_text) > 3000:
            metrics_text = metrics_text[:3000] + "\n... [truncated]"
        return metrics_text
    except Exception as e:
        return f"ERROR: {e}"


@tool
def post_agent_log(action: str, reasoning: str, imsi: str = "", status: str = "success") -> str:
    """
    Log an agent action and its reasoning to the audit trail.
    Always call this after taking any corrective action.
    Args:
        action: Short description of what was done
        reasoning: Why this action was taken
        imsi: The IMSI affected (optional)
        status: 'success', 'failure', or 'escalate'
    """
    try:
        payload = {
            "action": action,
            "reasoning": reasoning,
            "imsi": imsi if imsi else None,
            "status": status
        }
        r = requests.post(f"{NEF_BASE}/agent/log", json=payload, timeout=10)
        r.raise_for_status()
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return f"ERROR: {e}"


@tool
def get_agent_logs() -> str:
    """Retrieve the full agent audit trail — all past actions and reasoning."""
    try:
        r = requests.get(f"{NEF_BASE}/agent/log", timeout=10)
        r.raise_for_status()
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return f"ERROR: {e}"


@tool
def get_baseline_policies() -> str:
    """
    Return the known baseline (expected) QoS policies for all registered UEs.
    Compare these against current policies to detect violations.
    """
    return json.dumps(BASELINE_POLICIES, indent=2)


# ─────────────────────────────────────────────
# LangGraph State
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


# ─────────────────────────────────────────────
# Build the ReAct Graph
# ─────────────────────────────────────────────

tools = [
    get_core_status,
    get_active_sessions,
    get_subscriber_session,
    get_subscriber_policy,
    update_subscriber_ambr,
    get_metrics_snapshot,
    post_agent_log,
    get_agent_logs,
    get_baseline_policies,
]

llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE,
    temperature=0.1,
)
llm_with_tools = llm.bind_tools(tools)


SYSTEM_PROMPT = """You are an autonomous 5G QoS Orchestration Agent managing a live 5G core network (Ella-Core).
You are running Phase 6 validation — iperf3 traffic is being injected through UE tunnels to simulate policy violations.

Your mission per cycle:
1. OBSERVE  — get_core_status, get_active_sessions, get_metrics_snapshot
2. BASELINE — get_baseline_policies to know the EXPECTED policy per IMSI
3. COMPARE  — for each active IMSI, call get_subscriber_policy and compare to baseline
4. DETECT   — if current policy ≠ baseline policy, that is a VIOLATION
5. ACT      — call update_subscriber_ambr to restore the baseline policy
6. VERIFY   — call get_subscriber_policy again to confirm the correction took effect
7. LOG      — call post_agent_log with action, reasoning, imsi, and status

Violation rules:
- current policy = 'throttled' but baseline = 'default'  →  VIOLATION: restore to 'default'
- subscriber has NO policy assigned                        →  VIOLATION: assign 'default'
- subscriber is 'throttled' across 3+ consecutive cycles  →  log with status='escalate'
- core health degraded                                     →  log immediately

Decision rules:
- Only change policies that are WRONG — never touch correct ones
- Always log BEFORE and AFTER a corrective action
- If update_subscriber_ambr returns an error, log it with status='failure' and move on
- After all IMSIs are checked and corrections are made, output a summary starting with:
  "CYCLE COMPLETE: <brief summary of what was found and done>"

Be concise and systematic. Do not repeat tool calls unnecessarily.
"""


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def call_model(state: AgentState) -> AgentState:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile()


# ─────────────────────────────────────────────
# Rich helpers
# ─────────────────────────────────────────────

def log_tool_call(msg):
    if not hasattr(msg, "tool_calls") or not msg.tool_calls:
        return
    for tc in msg.tool_calls:
        args_str = json.dumps(tc.get("args", {}), indent=2)
        console.print(Panel(
            f"[bold cyan]{tc['name']}[/bold cyan]\n[dim]{args_str}[/dim]",
            title="[yellow]⚙ Tool Call[/yellow]",
            border_style="yellow",
            expand=False,
        ))


def log_tool_result(msg):
    if not isinstance(msg, ToolMessage):
        return
    content = msg.content
    display = content if len(content) <= 800 else content[:800] + "\n[dim]... truncated[/dim]"

    # Color-code by outcome
    if "ERROR" in content:
        style, icon = "red", "✘"
    elif "VIOLATION" in content.upper() or "throttled" in content.lower():
        style, icon = "yellow", "⚠"
    else:
        style, icon = "green", "✔"

    console.print(Panel(
        display,
        title=f"[bold {style}]{icon} Tool Result — {msg.name}[/bold {style}]",
        border_style=style,
        expand=False,
    ))


# ─────────────────────────────────────────────
# Cycle runner
# ─────────────────────────────────────────────

_violation_counts: dict = {}   # imsi → consecutive violation cycle count

def run_agent_cycle(cycle_number: int) -> str:
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    console.print()
    console.rule(
        f"[bold magenta]CYCLE {cycle_number}[/bold magenta]  [dim]{timestamp}[/dim]",
        style="magenta"
    )

    # Pre-cycle: local violation snapshot for display
    try:
        sessions_r = requests.get(f"{NEF_BASE}/core/sessions", timeout=5).json()
        active = sessions_r.get("active_sessions", [])
        if active:
            tbl = Table(title="Active Sessions", box=box.SIMPLE_HEAVY, show_header=True)
            tbl.add_column("IMSI", style="cyan")
            tbl.add_column("Registered")
            tbl.add_column("Policy")
            for s in active:
                imsi = s.get("imsi", "?")
                registered = "[green]Yes[/green]" if s.get("status", {}).get("registered") else "[red]No[/red]"
                policy = s.get("policyName") or "[red]NONE[/red]"
                expected = BASELINE_POLICIES.get(imsi, "default")
                if policy != expected:
                    policy = f"[red]{policy} ⚠ (expected: {expected})[/red]"
                    _violation_counts[imsi] = _violation_counts.get(imsi, 0) + 1
                else:
                    _violation_counts[imsi] = 0
                    policy = f"[green]{policy}[/green]"
                tbl.add_row(imsi, registered, policy)
            console.print(tbl)
    except Exception:
        pass

    trigger_message = HumanMessage(
        content=(
            f"[Cycle {cycle_number} — {timestamp}] "
            "Perform a full observe→baseline→compare→detect→act→verify→log cycle. "
            "Check all active subscriber policies against their baselines. "
            "Correct any violations. Output 'CYCLE COMPLETE:' with a summary when done."
        )
    )

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            trigger_message,
        ]
    }

    with Live(
        Spinner("dots", text="[cyan]Agent running ReAct loop...[/cyan]"),
        console=console,
        refresh_per_second=10,
        transient=True,
    ) as live:
        final_state = graph.invoke(initial_state, config={"recursion_limit": 60})
        live.stop()

    # Print message trace
    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                log_tool_call(msg)
            elif msg.content and "CYCLE COMPLETE" not in msg.content:
                console.print(Panel(
                    msg.content,
                    title="[bold blue]🤖 Agent Reasoning[/bold blue]",
                    border_style="blue",
                    expand=False,
                ))
        elif isinstance(msg, ToolMessage):
            log_tool_result(msg)

    summary = "No summary produced."
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            summary = msg.content
            break

    if "CYCLE COMPLETE" in summary:
        # Extract just the summary line for the panel
        summary_line = next(
            (l for l in summary.splitlines() if "CYCLE COMPLETE" in l),
            summary
        )
        console.print(Panel(
            summary_line,
            title="[bold green]✅ Cycle Complete[/bold green]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            summary[:500],
            title="[bold yellow]⚠ Cycle Ended (no CYCLE COMPLETE marker)[/bold yellow]",
            border_style="yellow",
        ))

    return summary


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    console.print(Panel(
        f"[bold]NEF Proxy    :[/bold] [cyan]{NEF_BASE}[/cyan]\n"
        f"[bold]LLM Model    :[/bold] [cyan]{LLM_MODEL}[/cyan] via Ollama ([dim]{OLLAMA_BASE}[/dim])\n"
        f"[bold]Interval     :[/bold] [cyan]{OBSERVE_INTERVAL_SECONDS}s[/cyan] between cycles\n"
        f"[bold]Baselines    :[/bold] {json.dumps(BASELINE_POLICIES)}\n"
        f"[bold]Phase        :[/bold] [magenta]6 — Validation[/magenta]",
        title="[bold magenta]5G QoS Orchestration Agent — Phase 6[/bold magenta]",
        border_style="magenta",
    ))

    boot_table = Table(title="Boot Checks", box=box.ROUNDED)
    boot_table.add_column("Service", style="bold")
    boot_table.add_column("Status")
    boot_table.add_column("Detail")

    try:
        r = requests.get(f"{NEF_BASE}/core/status", timeout=5)
        boot_table.add_row("NEF Proxy", "[green]✔ Reachable[/green]", f"HTTP {r.status_code}")
    except Exception as e:
        boot_table.add_row("NEF Proxy", "[red]✘ Unreachable[/red]", str(e))

    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        found = any(LLM_MODEL.split(":")[0] in m for m in models)
        boot_table.add_row(
            "Ollama",
            "[green]✔ Reachable[/green]" if found else "[yellow]⚠ Model missing[/yellow]",
            f"'{LLM_MODEL}' found" if found else f"Run: ollama pull {LLM_MODEL}",
        )
    except Exception as e:
        boot_table.add_row("Ollama", "[red]✘ Unreachable[/red]", str(e))

    console.print(boot_table)
    console.print()
    console.print("[dim]Tip: Run the traffic scenario script inside ella-ueransim to trigger violations.[/dim]\n")

    cycle = 1
    while cycle <= MAX_LOOP_ITERATIONS:
        try:
            run_agent_cycle(cycle)
        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted. Shutting down.[/bold red]")
            break
        except Exception as e:
            console.print(f"[red]Cycle {cycle} error:[/red] {e}")

        cycle += 1
        if cycle <= MAX_LOOP_ITERATIONS:
            console.print(
                f"\n[dim]Next cycle in [cyan]{OBSERVE_INTERVAL_SECONDS}s[/cyan]...[/dim]"
            )
            time.sleep(OBSERVE_INTERVAL_SECONDS)

    console.print(Rule("[bold red]Agent Stopped[/bold red]"))


if __name__ == "__main__":
    main()