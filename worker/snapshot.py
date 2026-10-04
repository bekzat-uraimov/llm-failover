import os
import json
import base64
import pickle
import uuid
from typing import Any, Dict

ROOT = os.path.dirname(__file__)
SNAPSHOT_DIR = os.path.join(ROOT, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def _encode_rng_state(rng_state: Any) -> str:
    return base64.b64encode(pickle.dumps(rng_state)).decode("ascii")


def _decode_rng_state(b64: str) -> Any:
    return pickle.loads(base64.b64decode(b64.encode("ascii")))


def save_snapshot(token_history, rng_state, position: int, metadata: Dict[str, Any] = None) -> str:
    """Save a snapshot and return the snapshot id."""
    sid = uuid.uuid4().hex
    # Accept either:
    # - a raw RNG state object (e.g. returned by random.getstate()) -> encode it
    # - a base64-encoded pickled RNG state string (already encoded by a client)
    if isinstance(rng_state, str):
        # validate that the provided string decodes to a pickled object
        try:
            _ = _decode_rng_state(rng_state)
        except Exception as e:
            raise ValueError("rng_state string is not a valid base64 pickled RNG state") from e
        rng_state_b64 = rng_state
    else:
        rng_state_b64 = _encode_rng_state(rng_state)

    data = {
        "token_history": list(token_history),
        "rng_state": rng_state_b64,
        "position": int(position),
        "metadata": metadata or {},
    }
    path = os.path.join(SNAPSHOT_DIR, f"{sid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return sid


def load_snapshot(sid: str):
    """Load a snapshot and return (token_history, rng_state, position, metadata)."""
    path = os.path.join(SNAPSHOT_DIR, f"{sid}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(sid)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    token_history = data["token_history"]
    rng_state = _decode_rng_state(data["rng_state"])
    position = int(data["position"])
    metadata = data.get("metadata", {})
    return token_history, rng_state, position, metadata


def list_snapshots():
    ids = []
    for fn in os.listdir(SNAPSHOT_DIR):
        if fn.endswith(".json"):
            ids.append(fn[:-5])
    return ids
