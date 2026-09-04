from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import httpx
import datetime
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = FastAPI(title="Ella-Core NEF Proxy", description="Agentic API for 5G QoS Orchestration")

# Configuration
ELLA_CORE_URL = os.getenv("ELLA_CORE_URL", "http://127.0.0.1:5002/api/v1")
# You must generate this via POST /api/v1/auth/login or the Ella UI
ELLA_API_TOKEN = os.getenv("ELLA_CORE_TOKEN")  # Replace with your actual token

headers = {
    "Authorization": f"Bearer {ELLA_API_TOKEN}",
    "Content-Type": "application/json"
}

# In-memory store for agent logs
agent_logs: List[Dict[str, Any]] = []

class LogEntry(BaseModel):
    action: str
    reasoning: str
    imsi: str = None
    status: str = "success"

@app.get("/core/status")
async def get_status():
    """Get core health and status."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{ELLA_CORE_URL}/status")
        resp.raise_for_status()
        return resp.json().get("result", {})

@app.get("/core/metrics")
async def get_metrics():
    """Fetch Prometheus metrics snapshot."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{ELLA_CORE_URL}/metrics")
        resp.raise_for_status()
        # Returns raw Prometheus text format
        return {"metrics": resp.text}

@app.get("/core/sessions")
async def get_active_sessions():
    """List all registered subscribers (active PDU sessions)."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{ELLA_CORE_URL}/subscribers", headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        
        data = resp.json().get("result", {}).get("items", [])
        # Filter only registered devices
        active_sessions = [sub for sub in data if sub.get("status", {}).get("registered")]
        return {"active_sessions": active_sessions}

@app.get("/core/sessions/{imsi}")
async def get_subscriber_session(imsi: str):
    """Get detailed session info for a specific UE."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{ELLA_CORE_URL}/subscribers/{imsi}", headers=headers)
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="Subscriber not found")
        resp.raise_for_status()
        return resp.json().get("result", {})

@app.get("/core/subscriber/{imsi}")
async def get_subscriber_profile(imsi: str):
    """Alias for session detail in Ella-Core."""
    return await get_subscriber_session(imsi)

@app.get("/core/policy/{imsi}")
async def get_subscriber_policy(imsi: str):
    """Fetch the QoS policy currently assigned to the subscriber."""
    sub_data = await get_subscriber_session(imsi)
    policy_name = sub_data.get("policyName")
    
    if not policy_name:
        raise HTTPException(status_code=404, detail="No policy assigned to this subscriber")

    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{ELLA_CORE_URL}/policies/{policy_name}", headers=headers)
        resp.raise_for_status()
        return {"imsi": imsi, "policy": resp.json().get("result", {})}

@app.put("/core/subscriber/{imsi}/ambr")
async def update_subscriber_ambr(imsi: str, body: dict = Body(...)):
    """
    Update subscriber's QoS policy.
    Expects: {"policy_name": "premium"} or {"target_policy": "premium"}
    """
    policy_name = body.get("policy_name") or body.get("target_policy")
    
    if not policy_name:
        raise HTTPException(status_code=400, detail="Missing 'policy_name' or 'target_policy' in request body")
    
    # Verify policy exists first
    async with httpx.AsyncClient() as client:
        policy_check = await client.get(f"{ELLA_CORE_URL}/policies/{policy_name}", headers=headers)
        if policy_check.status_code != 200:
            raise HTTPException(status_code=404, detail=f"Policy '{policy_name}' not found in Ella-Core")
    
    # Now update subscriber
    payload = {"policyName": policy_name}
    
    async with httpx.AsyncClient() as client:
        resp = await client.put(f"{ELLA_CORE_URL}/subscribers/{imsi}", json=payload, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Ella-Core error: {resp.text}")
        
        return {
            "message": f"Successfully updated IMSI {imsi} to policy '{policy_name}'",
            "imsi": imsi,
            "policy": policy_name
        }

@app.put("/core/subscriber/{imsi}/slice")
async def update_subscriber_slice(imsi: str, sst: int, sd: str = ""):
    """
    Placeholder for slice steering.
    Ella-Core configures slicing at the Operator level, not per-subscriber.
    """
    return {
        "warning": "Ella-Core OpenAPI does not support per-subscriber slice assignment.",
        "message": "Endpoint stubbed for agent compatibility."
    }

@app.post("/agent/log")
async def create_agent_log(entry: LogEntry):
    """Agent POSTs its reasoning and actions here for the audit trail."""
    log_record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        **entry.dict()
    }
    agent_logs.append(log_record)
    return {"message": "Log recorded"}

@app.get("/agent/log")
async def get_agent_logs():
    """Retrieve the agent's audit trail."""
    return {"logs": agent_logs}