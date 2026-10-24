"""Failover demo with real processes: the same story with and without killing its worker.

    python3 scripts/demo.py                   # kill -9 the generating worker at token 80
    python3 scripts/demo.py --signal stop     # freeze it instead; only the heartbeat can notice
    python3 scripts/demo.py --check           # several kill points, both signals, exit 1 on any mismatch

Starts two workers and a router, runs the story once untouched and once with the worker
interrupted partway, and compares the two outputs by SHA-256.
"""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "build", "worker")
MODEL = os.path.join(ROOT, "models", "stories15M.bin")
TOKENIZER = os.path.join(ROOT, "models", "tokenizer.bin")

SIGNALS = {"kill": signal.SIGKILL, "stop": signal.SIGSTOP}
RED, GREEN, DIM, RESET = "\033[31m", "\033[32m", "\033[2m", "\033[0m"


class Cluster:
    """Two worker processes and a router, all on free localhost ports."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.events_path = os.path.join(self.tmp.name, "events.jsonl")
        self.workers: dict[int, subprocess.Popen] = {}
        for _ in range(2):
            proc, port = self.start_worker(0)
            self.workers[port] = proc
        cmd = [sys.executable, "-m", "router.router", "--port", "0", "--events", self.events_path]
        for port in self.workers:
            cmd += ["--worker", f"127.0.0.1:{port}"]
        self.router = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.port = int(self.router.stdout.readline().rsplit(":", 1)[1])
        for port in self.workers:
            self.wait_ready(port)

    @staticmethod
    def start_worker(port: int) -> tuple[subprocess.Popen, int]:
        proc = subprocess.Popen(
            [WORKER, "--model", MODEL, "--tokenizer", TOKENIZER, "--port", str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        return proc, int(proc.stdout.readline().rsplit(":", 1)[1])

    def events(self) -> list[dict]:
        with open(self.events_path) as f:
            return [json.loads(line) for line in f]

    def wait_ready(self, port: int, timeout: float = 10.0):
        name = f"127.0.0.1:{port}"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = None
            for e in self.events():
                if e.get("worker") == name and e["event"] in ("worker_ready", "worker_lost"):
                    state = e["event"]
            if state == "worker_ready":
                return
            time.sleep(0.05)
        raise RuntimeError(f"worker {name} never became ready")

    def generating_worker(self) -> int:
        assigns = [e for e in self.events() if e["event"] == "assign"]
        return int(assigns[-1]["worker"].rsplit(":", 1)[1])

    def heal(self, port: int, sig: int):
        """Bring an interrupted worker back so the next run has two workers again."""
        if sig == signal.SIGSTOP:
            self.workers[port].send_signal(signal.SIGCONT)
        else:
            self.workers[port].wait()
            self.workers[port].stdout.close()
            self.workers[port], _ = self.start_worker(port)
        self.wait_ready(port)

    def close(self):
        for proc in [*self.workers.values(), self.router]:
            proc.send_signal(signal.SIGCONT)
            proc.kill()
            proc.wait()
            proc.stdout.close()
        self.tmp.cleanup()


def run_story(cluster: Cluster, args, kill_at: int | None = None, sig: int = signal.SIGKILL,
              echo: bool = False) -> dict:
    req = {"type": "GENERATE", "prompt": args.prompt, "seed": args.seed, "steps": args.steps}
    out = {"text": "", "victim": None, "pause_ms": None}
    killed_at = None
    with socket.create_connection(("127.0.0.1", cluster.port), timeout=30) as sock:
        f = sock.makefile("rwb")
        f.write(json.dumps(req).encode() + b"\n")
        f.flush()
        for line in f:
            msg = json.loads(line)
            if msg["type"] == "ERROR":
                raise RuntimeError(msg["error"])
            if msg["type"] == "END":
                break
            if killed_at and out["pause_ms"] is None:
                out["pause_ms"] = (time.monotonic() - killed_at) * 1000
            out["text"] += msg["text"]
            if echo:
                sys.stdout.write(msg["text"])
                sys.stdout.flush()
            if kill_at is not None and out["victim"] is None and msg["pos"] >= kill_at:
                out["victim"] = cluster.generating_worker()
                cluster.workers[out["victim"]].send_signal(sig)
                killed_at = time.monotonic()
                out["killed_pos"] = msg["pos"]
                if echo:
                    verb = "kill -9" if sig == signal.SIGKILL else "kill -STOP"
                    sys.stdout.write(f"{RED} [{verb} worker :{out['victim']}] {RESET}")
                    sys.stdout.flush()
    if echo:
        sys.stdout.write("\n")
    if kill_at is not None and out["victim"] is None:
        raise RuntimeError(f"story ended before token {kill_at}; pick an earlier kill point")
    resumed = [e for e in cluster.events() if e["event"] == "resumed"]
    if out["victim"] and resumed:
        out["resumed"] = resumed[-1]
    return out


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def demo(cluster: Cluster, args) -> bool:
    print(f"{DIM}normal run: seed {args.seed}, {args.steps} steps, nothing killed{RESET}")
    normal = run_story(cluster, args)
    print(f"{DIM}failover run: same request, worker interrupted at token {args.kill_at}{RESET}\n")
    failover = run_story(cluster, args, args.kill_at, SIGNALS[args.signal], echo=True)

    r = failover["resumed"]
    print(f"\n{DIM}worker :{failover['victim']} lost at token {failover['killed_pos']}, "
          f"resumed on {r['worker']}: replayed {r['pos'] - 1} tokens in {r['replay_ms']:.0f} ms, "
          f"client paused {failover['pause_ms']:.0f} ms{RESET}\n")
    a, b = sha(normal["text"]), sha(failover["text"])
    print(f"normal    sha256 {a}")
    print(f"failover  sha256 {b}")
    if a == b:
        print(f"{GREEN}identical, byte for byte{RESET}")
    else:
        print(f"{RED}DIFFERENT{RESET}")
    return a == b


def check(cluster: Cluster, args) -> bool:
    normal = sha(run_story(cluster, args)["text"])
    ok = True
    for name, sig in SIGNALS.items():
        for kill_at in (10, 80, 150):
            run = run_story(cluster, args, kill_at, sig)
            match = sha(run["text"]) == normal
            ok &= match
            print(f"{'ok  ' if match else 'FAIL'} {name:4} at token {kill_at:3}: worker :{run['victim']} -> "
                  f"{run['resumed']['worker']}, pause {run['pause_ms']:.0f} ms")
            cluster.heal(run["victim"], sig)
    print(f"normal run sha256 {normal}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--kill-at", type=int, default=80, help="interrupt the worker once this token streams")
    p.add_argument("--signal", choices=SIGNALS, default="kill")
    p.add_argument("--check", action="store_true", help="run every kill point and signal; exit 1 on a mismatch")
    args = p.parse_args()

    if not (os.path.exists(WORKER) and os.path.exists(MODEL)):
        sys.exit("run `make` and `make model` first")
    cluster = Cluster()
    try:
        ok = check(cluster, args) if args.check else demo(cluster, args)
    finally:
        cluster.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
