from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import joblib
from collections import deque
from datetime import datetime

# define the request envelope
class TrafficData(BaseModel):
    features: list

# define the enforcement report envelope
class BlockData(BaseModel):
    mac_address: str
    source_ip: str
    anomaly_score: float
    reason: str = "threshold exceeded"

# envelopes for manual admin actions queued by the dashboard.
# the controller polls /admin/pending_actions and runs the OVS
# commands for each one, so the API never touches the switch directly.
class ManualBlockRequest(BaseModel):
    source_ip: str
    reason: str = "manual block from dashboard"

class ManualUnblockRequest(BaseModel):
    mac_address: str

app = FastAPI(title="NGFW Inference Engine")

# load model artifacts via the Docker volume mount
scaler = joblib.load('/app/model/scaler.pkl')
top_features_idx = np.load('/app/model/top_features_idx.npy')
ort_session = ort.InferenceSession('/app/model/ngfw_ae.onnx')

# in-memory store for recent scan results.
# using a deque so it automatically drops old entries once it hits the limit.
# 200 is enough for the dashboard to show a meaningful history without
# eating memory on a long-running container.
MAX_ALERTS = 200
recent_alerts = deque(maxlen=MAX_ALERTS)

# in-memory store for enforcement actions.
# tracks which hosts have been blocked and why.
MAX_BLOCKS = 100
blocked_hosts = deque(maxlen=MAX_BLOCKS)

# in-memory queue of pending admin actions from the dashboard.
# the controller polls /admin/pending_actions periodically, drains this
# queue, and runs the corresponding OVS command for each action.
MAX_PENDING_ACTIONS = 50
pending_actions = deque(maxlen=MAX_PENDING_ACTIONS)

# simple monotonic action-id counter so each queued action is traceable
# in the controller logs. wrapped in a dict so it can be mutated from
# inside the endpoint functions without needing 'global'.
_action_id = {"n": 0}


@app.post("/predict")
def predict(data: TrafficData):
    # convert incoming list into a 2D numpy array
    raw_features = np.array(data.features).reshape(1, -1)

    # 1. ensure exactly 77 features for the scaler
    if raw_features.shape[1] > 77:
        raw_features = raw_features[:, :77]
    elif raw_features.shape[1] < 77:
        padded = np.zeros((1, 77))
        padded[0, :raw_features.shape[1]] = raw_features
        raw_features = padded

    # 2. scale the full feature set first
    scaled_full_features = scaler.transform(raw_features)

    # 3. extract only the 40 critical features using the random forest index
    final_features = scaled_full_features[:, top_features_idx].astype(np.float32)

    # 4. high-speed C++ ONNX inference
    inputs = {ort_session.get_inputs()[0].name: final_features}
    reconstruction = ort_session.run(None, inputs)[0]

    # 5. calculate mean squared error (anomaly score)
    mse = np.mean(np.square(final_features - reconstruction))
    is_threat = bool(mse > 0.001936)

    # 6. log the result so the dashboard can pick it up.
    # storing the raw 77 features so the SHAP tab can pull them
    # and run a local explanation without the user typing anything.
    alert_entry = {
        "timestamp": datetime.now().isoformat(),
        "threat_detected": is_threat,
        "malicious_score": float(mse),
        "feature_count": int(raw_features.shape[1]),
        "features": raw_features.flatten().tolist(),
    }
    recent_alerts.append(alert_entry)

    return {"threat_detected": is_threat, "malicious_score": float(mse)}


@app.get("/alerts")
def get_alerts():
    """Return the most recent scan results for the dashboard.
    Results are ordered newest-first so the dashboard can display
    the latest alerts at the top of the list."""
    return {"alerts": list(reversed(recent_alerts))}


@app.post("/block")
def report_block(data: BlockData):
    """Log an enforcement action from the controller.
    Called by the controller after it injects a drop rule into OVS,
    so the dashboard can display which hosts have been blocked."""
    block_entry = {
        "timestamp": datetime.now().isoformat(),
        "mac_address": data.mac_address,
        "source_ip": data.source_ip,
        "anomaly_score": data.anomaly_score,
        "reason": data.reason,
    }
    blocked_hosts.append(block_entry)
    print(f"[BLOCK] logged enforcement: {data.source_ip} ({data.mac_address})")
    return {"status": "logged", "entry": block_entry}


@app.get("/blocked")
def get_blocked():
    """Return the list of blocked hosts for the dashboard.
    Results are ordered newest-first."""
    return {"blocked": list(reversed(blocked_hosts))}


# ---------------------------------------------------------------------------
# admin endpoints for manual enforcement from the dashboard
# ---------------------------------------------------------------------------

@app.post("/admin/block")
def admin_block(data: ManualBlockRequest):
    """Queue a manual block action for the controller to execute.

    The controller polls /admin/pending_actions on a short interval,
    drains the queue, and runs ovs-ofctl for each action. The API
    itself never touches the switch -- it only records intent."""
    _action_id["n"] += 1
    action = {
        "id": _action_id["n"],
        "action": "block",
        "source_ip": data.source_ip,
        "reason": data.reason,
        "created_at": datetime.now().isoformat(),
    }
    pending_actions.append(action)
    return {"status": "queued", "action_id": action["id"]}


@app.post("/admin/unblock")
def admin_unblock(data: ManualUnblockRequest):
    """Queue a manual unblock action for the controller to execute.

    We also remove the host from the blocked list immediately so the
    dashboard reflects the intended state without waiting a poll cycle.
    If the controller later fails to remove the OVS rule, the host will
    re-appear in the blocked list on the next detection anyway."""
    _action_id["n"] += 1
    action = {
        "id": _action_id["n"],
        "action": "unblock",
        "mac_address": data.mac_address,
        "created_at": datetime.now().isoformat(),
    }
    pending_actions.append(action)

    # filter the blocked list. deque doesn't support predicate removal
    # so we rebuild it, preserving newest-first ordering.
    remaining = [b for b in blocked_hosts
                 if b.get("mac_address") != data.mac_address]
    blocked_hosts.clear()
    for b in remaining:
        blocked_hosts.append(b)

    return {"status": "queued", "action_id": action["id"]}


@app.get("/admin/pending_actions")
def get_pending_actions():
    """Return pending admin actions for the controller to execute.

    This is a destructive read: actions are drained from the queue
    once returned. The controller consumes the list in order."""
    actions = list(pending_actions)
    pending_actions.clear()
    return {"actions": actions}