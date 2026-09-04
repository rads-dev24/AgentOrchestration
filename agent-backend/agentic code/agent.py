"""
5G QoS Orchestration Agent — Phase 5
ReAct agent using LangGraph + Ollama (local LLM with tool calling)
Runs on Windows, connects to NEF Proxy at 100.92.250.111:8888
"""

import json
import time
import datetime
from typing import Annotated, TypedDict, List

import requests
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
NEF_BASE = "http://127.0.0.1:8000"
OLLAMA_BASE = "http://localhost:11434"   # Ollama running on Windows host

# qwen2.5:7b is well-tested for tool calling via Ollama
# alternatives: llama3.1:8b, mistral-nemo, mistral:7b-instruct
LLM_MODEL = "qwen2.5:7b"

OBSERVE_INTERVAL_SECONDS = 30   # how often the agent polls the network
MAX_LOOP_ITERATIONS = 100       # safety cap on the agentic loop


# ─────────────────────────────────────────────
# Tool definitions — each maps to a NEF proxy endpoint
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
        # Return first 3000 chars to avoid overwhelming the LLM context
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
        status: 'success' or 'failure'
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
]

# Bind tools to the LLM
llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE,
    temperature=0.1,      # low temp for deterministic network decisions
)
llm_with_tools = llm.bind_tools(tools)


SYSTEM_PROMPT = """You are an autonomous 5G QoS Orchestration Agent managing a live 5G core network (Ella-Core).

Your mission:
1. OBSERVE: Poll core status, active sessions, and subscriber policies
2. REASON: Identify policy violations, anomalies, or degraded subscribers
3. ACT: Issue corrective actions (AMBR updates, policy reassignment)
4. VERIFY: Re-read subscriber state to confirm changes took effect
5. LOG: Always call post_agent_log after any corrective action

QoS Rules you must enforce:
- Every registered subscriber MUST have a policy assigned
- If a subscriber has no policy, assign 'default'
- If a subscriber's policy is 'throttled' AND they have been throttled for more than 2 audit log entries, escalate to human by logging with status='escalate'
- Report any core health degradation immediately in the log

Decision framework:
- Start each cycle by checking core status and active sessions
- For each active session, verify the subscriber has a valid policy
- Take the minimum necessary action — don't change policies that are already correct
- Always log your reasoning before AND after acting

You have access to these tools:
- get_core_status: Check if the 5G core is healthy
- get_active_sessions: List all registered UEs
- get_subscriber_session: Detailed info for one UE
- get_subscriber_policy: Current QoS policy for one UE
- update_subscriber_ambr: Change the QoS policy for a UE
- get_metrics_snapshot: Prometheus metrics from Ella-Core
- post_agent_log: Write to the audit trail
- get_agent_logs: Read the audit trail

Think step by step. Be concise. After completing one full observe→reason→act→verify cycle, output a short summary starting with "CYCLE COMPLETE:" so the orchestrator knows you are done.
"""


def should_continue(state: AgentState) -> str:
    """Route: if the last message has tool calls → run tools, else → end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def call_model(state: AgentState) -> AgentState:
    """Invoke the LLM with current message history."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Build graph
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()


# ─────────────────────────────────────────────
# Orchestration Loop
# ─────────────────────────────────────────────

def run_agent_cycle(cycle_number: int) -> str:
    """Run one full observe→reason→act→verify cycle."""
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    print(f"\n{'='*60}")
    print(f"  CYCLE {cycle_number} — {timestamp}")
    print(f"{'='*60}")

    trigger_message = HumanMessage(
        content=f"[Cycle {cycle_number} — {timestamp}] "
                "Perform a full observe→reason→act→verify cycle on the 5G network. "
                "Check core health, inspect all active sessions, enforce QoS policies, "
                "and log all actions. Output 'CYCLE COMPLETE:' with a summary when done."
    )

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            trigger_message,
        ]
    }

    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": 50}
    )

    # Extract last assistant message
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            print(f"\n[Agent Summary]\n{msg.content}\n")
            return msg.content

    return "No summary produced."


def main():
    print("=" * 60)
    print("  5G QoS Orchestration Agent — Phase 5")
    print(f"  NEF Proxy : {NEF_BASE}")
    print(f"  LLM Model : {LLM_MODEL} via Ollama ({OLLAMA_BASE})")
    print(f"  Interval  : {OBSERVE_INTERVAL_SECONDS}s between cycles")
    print("=" * 60)

    # Quick connectivity check
    print("\n[Boot] Checking NEF proxy connectivity...")
    try:
        r = requests.get(f"{NEF_BASE}/core/status", timeout=5)
        print(f"[Boot] NEF Proxy reachable — HTTP {r.status_code}")
    except Exception as e:
        print(f"[Boot] WARNING: Cannot reach NEF proxy: {e}")
        print("       Ensure the VM is running and NEF proxy is started on port 8888.")

    print("\n[Boot] Checking Ollama connectivity...")
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"[Boot] Ollama reachable — available models: {models}")
        if not any(LLM_MODEL.split(":")[0] in m for m in models):
            print(f"[Boot] WARNING: Model '{LLM_MODEL}' not found locally.")
            print(f"       Run: ollama pull {LLM_MODEL}")
    except Exception as e:
        print(f"[Boot] WARNING: Cannot reach Ollama: {e}")

    print("\n[Boot] Starting agentic loop...\n")

    cycle = 1
    while cycle <= MAX_LOOP_ITERATIONS:
        try:
            run_agent_cycle(cycle)
        except KeyboardInterrupt:
            print("\n[Agent] Interrupted by user. Shutting down.")
            break
        except Exception as e:
            print(f"[Agent] Cycle {cycle} error: {e}")

        cycle += 1
        if cycle <= MAX_LOOP_ITERATIONS:
            print(f"[Agent] Sleeping {OBSERVE_INTERVAL_SECONDS}s until next cycle...")
            time.sleep(OBSERVE_INTERVAL_SECONDS)

    print("[Agent] Max iterations reached. Exiting.")


if __name__ == "__main__":
    main()