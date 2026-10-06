import json
import os
import random
import re
import tempfile
import uuid

SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", os.path.join(os.path.dirname(__file__), "snapshots"))

_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _path(sid: str) -> str:
    # sid comes straight from HTTP requests; anything but a uuid4 hex could escape SNAPSHOT_DIR
    if not isinstance(sid, str) or not _ID_RE.match(sid):
        raise ValueError(f"invalid snapshot id: {sid!r}")
    return os.path.join(SNAPSHOT_DIR, f"{sid}.json")


def parse_rng_state(state) -> tuple:
    """Turn a JSON-decoded random.getstate() back into the tuple form setstate() wants.

    Raises ValueError if it isn't a valid Mersenne Twister state.
    """
    try:
        version, internal, gauss = state
        parsed = (version, tuple(internal), gauss)
        random.Random().setstate(parsed)
    except (TypeError, ValueError) as e:
        raise ValueError("invalid rng_state") from e
    return parsed


def save_snapshot(token_history: list[int], rng_state, position: int, metadata: dict | None = None) -> str:
    if not all(isinstance(t, int) for t in token_history):
        raise ValueError("token_history must be a list of ints")
    if not isinstance(position, int) or not 0 <= position <= len(token_history):
        raise ValueError("position must be an int between 0 and len(token_history)")
    rng_state = parse_rng_state(rng_state)

    sid = uuid.uuid4().hex
    data = {
        "token_history": list(token_history),
        "rng_state": rng_state,
        "position": position,
        "metadata": metadata or {},
    }

    # Write to a temp file and rename, so a crash mid-write never leaves a half-written snapshot
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=SNAPSHOT_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, _path(sid))
    except BaseException:
        os.unlink(tmp)
        raise
    return sid


def load_snapshot(sid: str) -> tuple[list[int], tuple, int, dict]:
    """Return (token_history, rng_state, position, metadata). Raises FileNotFoundError if missing."""
    with open(_path(sid), encoding="utf-8") as f:
        data = json.load(f)
    return data["token_history"], parse_rng_state(data["rng_state"]), data["position"], data["metadata"]


def list_snapshots() -> list[str]:
    if not os.path.isdir(SNAPSHOT_DIR):
        return []
    return sorted(fn[:-5] for fn in os.listdir(SNAPSHOT_DIR) if fn.endswith(".json"))
