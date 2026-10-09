# llm-failover

Kill the worker mid-sentence. Another one finishes it, byte for byte.

When an LLM worker dies halfway through a generation, its KV cache dies with it. This project takes the position that the KV cache doesn't need to survive. Given the model weights, the token history, the sampler's RNG state and the current position, the next token is fully determined. So a replacement worker can replay the tokens, rebuild the cache and continue the exact same output. This repo builds that recovery path one piece at a time, using Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) `stories15M` model as the workload.

## Status

Done:

- **Determinism check.** Ran llama2.c twice with `-s 42` and got identical output; `-s 43` diverged. This was a manual check.
- **Snapshots.** [worker/snapshot.py](worker/snapshot.py) saves and loads `(token_history, rng_state, position, metadata)` as plain JSON. Writes are atomic (temp file + rename), so a crash mid-save can't leave a corrupt snapshot.
- **Worker API.** [worker/server.py](worker/server.py) exposes snapshot save/load/inspect over HTTP and validates every input at the boundary.
- **Router.** [router/server.py](router/server.py) proxies snapshot calls to a worker and returns 502/504 when the worker is unreachable or times out.
- **Resume test.** Generate 100 tokens, snapshot at 50, restore into a fresh process, continue. The output matches the uninterrupted run. The test uses Python's `random` as a stand-in for the sampler. Hooking up the real model is next.

Not built yet:

- Worker that runs the actual model and streams tokens
- Router-driven failover: heartbeats, detecting a dead worker, resuming the session on another one
- Epoch fencing so a worker that comes back late can't write stale output
- Client that reconnects to a stream without seeing duplicate or missing tokens

## Run it

Requires Python 3.10+. No dependencies outside the standard library.

```bash
make test      # 16 tests
make worker    # worker on :8001
make router    # router on :8002, forwards to $WORKER_URL (default http://localhost:8001)
```

Snapshots go to `worker/snapshots/` unless you set `SNAPSHOT_DIR`.

## API

All endpoints are on both the worker and the router.

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| POST | `/snapshot/save` | `{"token_history", "rng_state", "position"?, "metadata"?}` | `{"id"}` |
| POST | `/snapshot/load` | `{"id"}` | full snapshot, including `rng_state` |
| GET | `/snapshot/<id>` | | position, token count, metadata |
| GET | `/snapshot` | | list of snapshot ids |

`rng_state` is Python's `random.getstate()` serialized as JSON. Example:

```bash
STATE=$(python3 -c 'import json, random; print(json.dumps(random.Random(42).getstate()))')
curl -s -X POST localhost:8002/snapshot/save -d "{\"token_history\": [1, 2, 3], \"rng_state\": $STATE}"
# {"id": "81e4a0a8a9d94506b09ba437505de015"}
```

## Design notes

- **Replay instead of shipping the KV cache.** For stories15M the full KV cache is about 3.5 MB, while 256 tokens of history is about 1 KB. Replaying costs compute on recovery but keeps the durable state tiny.
- **JSON, not pickle.** An earlier version stored RNG state with `pickle`, which meant anyone who could reach the worker could run code on it. The state is just integers, so JSON is enough.
- **Snapshot ids are checked before they touch the filesystem.** Only 32-char hex ids are accepted, which blocks path traversal like `../../etc/passwd`.

## Credits

Model and reference runtime: Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c).
