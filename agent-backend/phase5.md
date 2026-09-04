# Phase 5 — AI QoS Orchestration Agent

## Overview

A LangGraph ReAct agent that autonomously enforces QoS policies on a live 5G core network.
Runs on your **Windows host**, talks to the **NEF Proxy** on the VM at `192.168.56.11:8888`.

```
Windows Host
  └── agent.py (LangGraph ReAct loop)
        ├── Ollama (local LLM, qwen2.5:7b)
        └── NEF Proxy (192.168.56.11:8888)
              └── Ella-Core REST API
```

---

## Prerequisites

### 1. Ollama (Windows)

Install Ollama from https://ollama.com/download

Pull the recommended model (supports tool calling):
```bash
ollama pull qwen2.5:7b
```

**Alternatives with tool calling support:**
| Model | Size | Notes |
|-------|------|-------|
| `qwen2.5:7b` | ~4.7 GB | ✅ Recommended — excellent tool use |
| `qwen2.5:14b` | ~9 GB | Better reasoning, needs 16 GB RAM |
| `llama3.1:8b` | ~4.7 GB | Good tool calling, Meta model |
| `mistral-nemo` | ~7 GB | Strong function calling |

### 2. NEF Proxy (VM)

Ensure the NEF Proxy from Phase 3 is running on the VM:
```bash
# On VM: 192.168.56.11
cd /path/to/nef-proxy
uvicorn NEF-Proxy:app --host 0.0.0.0 --port 8888
```

Test from Windows:
```
curl http://192.168.56.11:8888/core/status
```

### 3. Python Environment (Windows)

```bash
# Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# or: source venv/bin/activate  (if using WSL/Git Bash)

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Agent

```bash
python agent.py
```

**Sample boot output:**
```
============================================================
  5G QoS Orchestration Agent — Phase 5
  NEF Proxy : http://192.168.56.11:8888
  LLM Model : qwen2.5:7b via Ollama (http://localhost:11434)
  Interval  : 30s between cycles
============================================================

[Boot] Checking NEF proxy connectivity...
[Boot] NEF Proxy reachable — HTTP 200
[Boot] Checking Ollama connectivity...
[Boot] Ollama reachable — available models: ['qwen2.5:7b']
[Boot] Starting agentic loop...

============================================================
  CYCLE 1 — 2025-01-15T10:00:00Z
============================================================
```

---

## Agent Behavior

Each cycle follows the **Observe → Reason → Act → Verify** loop:

1. **Observe**: `get_core_status` + `get_active_sessions`
2. **Reason**: LLM evaluates sessions against QoS rules
3. **Act**: `update_subscriber_ambr` if policy violation detected
4. **Verify**: `get_subscriber_policy` to confirm change
5. **Log**: `post_agent_log` with action + reasoning

### QoS Rules Enforced

| Condition | Action |
|-----------|--------|
| Subscriber has no policy | Assign `default` policy |
| Subscriber repeatedly throttled (2+ log entries) | Log with `status=escalate` |
| Core health degraded | Log warning immediately |

---

## Configuration

Edit the top of `agent.py` to change:

```python
NEF_BASE = "http://192.168.56.11:8888"   # VM address
OLLAMA_BASE = "http://localhost:11434"    # Ollama on Windows
LLM_MODEL = "qwen2.5:7b"                 # Model to use
OBSERVE_INTERVAL_SECONDS = 30            # Poll frequency
MAX_LOOP_ITERATIONS = 100                # Safety cap
```

---

## Viewing Audit Logs

The agent writes every action to the NEF Proxy's in-memory audit trail:

```bash
curl http://192.168.56.11:8888/agent/log
```

Or from Python:
```python
import requests
logs = requests.get("http://192.168.56.11:8888/agent/log").json()
for entry in logs["logs"]:
    print(entry)
```

---

## Troubleshooting

**"Model not found"**
```bash
ollama pull qwen2.5:7b
```

**"Cannot reach NEF proxy"**
- Check VM is running: `vagrant status`
- Check proxy is started on the VM
- Verify host-only adapter IP: `192.168.56.11`

**Tool calls not being made**
- Ensure you're using a model that supports tool calling (see table above)
- `qwen2.5:7b` is the most reliable choice with LangChain Ollama

**High CPU / slow responses**
- Use a smaller model: `qwen2.5:3b`
- Increase `OBSERVE_INTERVAL_SECONDS`
- Reduce `recursion_limit` in `graph.invoke()`