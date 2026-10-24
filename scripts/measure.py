"""Measure throughput, failover pause and replay cost; print the results as markdown tables.

    python3 scripts/measure.py              # about 3 minutes
    python3 scripts/measure.py --runs 1     # quicker, noisier

Every number comes from real processes on localhost: C++ workers, the router, and a
client on the other end of a socket.
"""

import argparse
import json
import platform
import signal
import socket
import statistics
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from demo import MODEL, Cluster, run_story

STORY = argparse.Namespace(prompt="Once upon a time", seed=42, steps=256)


def machine() -> str:
    if sys.platform == "darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
    else:
        cpu = next((line.split(":", 1)[1] for line in open("/proc/cpuinfo") if line.startswith("model name")), "")
    cc = subprocess.run(["c++", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    system = "macOS" if sys.platform == "darwin" else platform.system()
    return f"{cpu.strip() or platform.machine()}, {system}, {cc}, Python {platform.python_version()}"


def prompt_tokens(cluster: Cluster) -> int:
    return [e for e in cluster.events() if e["event"] == "started"][-1]["prompt_tokens"]


def generated(run: dict, n_prompt: int) -> list[tuple[float, int]]:
    return [(t, pos) for t, pos in run["times"] if pos >= n_prompt]


def throughput(runs: int) -> list[str]:
    cluster = Cluster(1)
    try:
        rates, gaps = [], []
        for _ in range(runs):
            run = run_story(cluster, STORY)
            gen = generated(run, prompt_tokens(cluster))
            rates.append((len(gen) - 1) / (gen[-1][0] - gen[0][0]))
            gaps += [(b[0] - a[0]) * 1000 for a, b in zip(gen, gen[1:])]
    finally:
        cluster.close()
    gaps.sort()
    return [
        "| One session, through the router | |",
        "| --- | --- |",
        f"| tokens per second | {statistics.median(rates):.0f} |",
        f"| time between tokens, median | {statistics.median(gaps):.1f} ms |",
        f"| time between tokens, p99 | {gaps[int(len(gaps) * 0.99)]:.1f} ms |",
    ]


def concurrency(runs: int) -> list[str]:
    rows = ["| Sessions at once | 1 worker | 2 workers |", "| --- | --- | --- |"]
    results: dict[tuple[int, int], float] = {}
    for n_workers in (1, 2):
        cluster = Cluster(n_workers)
        try:
            for sessions in (1, 2, 4):
                rates = []
                for _ in range(runs):
                    start = time.monotonic()
                    with ThreadPoolExecutor(sessions) as pool:
                        done = list(pool.map(lambda _: run_story(cluster, STORY), range(sessions)))
                    wall = time.monotonic() - start
                    n_prompt = prompt_tokens(cluster)
                    rates.append(sum(len(generated(r, n_prompt)) for r in done) / wall)
                results[(sessions, n_workers)] = statistics.median(rates)
        finally:
            cluster.close()
    for sessions in (1, 2, 4):
        rows.append(f"| {sessions} | {results[(sessions, 1)]:.0f} tok/s | {results[(sessions, 2)]:.0f} tok/s |")
    return rows


def failover(runs: int) -> list[str]:
    rows = ["| Worker interrupted at token | `kill -9`: pause seen by client | `kill -STOP`: pause seen by client |",
            "| --- | --- | --- |"]
    cluster = Cluster(2)
    try:
        for kill_at in (64, 128, 192):
            cells = []
            for sig in (signal.SIGKILL, signal.SIGSTOP):
                pauses = []
                for _ in range(runs):
                    run = run_story(cluster, STORY, kill_at, sig)
                    pauses.append(run["pause_ms"])
                    cluster.heal(run["victim"], sig)
                cells.append(f"{statistics.median(pauses):.0f} ms")
            rows.append(f"| {kill_at} | {cells[0]} | {cells[1]} |")
    finally:
        cluster.close()
    return rows


def long_history(port: int, n: int) -> tuple[list[int], int]:
    """A real token history of at least n tokens, from whichever seed first gets that far."""
    with socket.create_connection(("127.0.0.1", port)) as sock:
        f = sock.makefile("rwb")
        for seed in range(1, 100):
            f.write(json.dumps({"type": "START", "session": "m", "epoch": 1, "prompt": "", "seed": seed}).encode() + b"\n")
            f.flush()
            tokens, rng = [], seed
            for line in f:
                msg = json.loads(line)
                if msg["type"] == "STARTED":
                    tokens = msg["tokens"]
                elif msg["type"] == "TOKEN":
                    tokens.append(msg["token"])
                    rng = msg["rng_state"]
                elif msg["type"] == "DONE":
                    break
            if len(tokens) >= n:
                return tokens, rng
    raise RuntimeError(f"no story reached {n} tokens")


def replay_cost(runs: int) -> list[str]:
    proc, port = Cluster.start_worker(0)
    try:
        tokens, rng = long_history(port, 256)
        rows = ["| Context length | Replay time | Per token | RESUME message |", "| --- | --- | --- | --- |"]
        with socket.create_connection(("127.0.0.1", port)) as sock:
            f = sock.makefile("rwb")
            for n in (64, 128, 256):
                resume = {"type": "RESUME", "session": "m", "epoch": 2, "tokens": tokens[:n], "rng_state": rng}
                line = json.dumps(resume).encode() + b"\n"
                times = []
                for _ in range(max(runs, 3)):
                    f.write(line)
                    f.flush()
                    for reply in f:
                        msg = json.loads(reply)
                        if msg["type"] == "RESUMED":
                            times.append(msg["replay_ms"])
                            # Only stop once the replay is done; a STOP mid-replay cancels it
                            f.write(b'{"type": "STOP", "session": "m"}\n')
                            f.flush()
                        if msg["type"] == "STOPPED":
                            break
                ms = statistics.median(times)
                rows.append(f"| {n} tokens | {ms:.0f} ms | {ms / (n - 1):.2f} ms | {len(line):,} bytes |")
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()
    return rows


def state_size() -> list[str]:
    with open(MODEL, "rb") as f:
        dim, _, n_layers, n_heads, n_kv_heads, _, seq_len = struct.unpack("7i", f.read(28))
    kv_bytes = n_layers * seq_len * (dim * n_kv_heads // n_heads) * 2 * 4
    log_bytes = seq_len * (4 + 8)  # token id + rng_state per committed token
    resume_bytes = seq_len * 4 + 8  # tokens + the last rng_state
    return [
        "| State for a full 256-token context | Size |",
        "| --- | --- |",
        f"| KV cache ({n_layers} layers × {seq_len} positions × {dim} dims × K and V × 4 bytes) | {kv_bytes / 1e6:.1f} MB |",
        f"| Router log (token id + RNG state per token) | {log_bytes / 1e3:.1f} KB |",
        f"| What a RESUME needs (tokens + the last RNG state) | {resume_bytes / 1e3:.1f} KB |",
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=int, default=3, help="repeat each measurement and report the median")
    args = p.parse_args()

    print(f"Measured on {machine()}. Median of {args.runs} runs. Story: seed 42, \"Once upon a time\".\n")
    for section in (throughput, concurrency, failover, replay_cost):
        print("\n".join(section(args.runs)) + "\n", flush=True)
    print("\n".join(state_size()))


if __name__ == "__main__":
    main()
