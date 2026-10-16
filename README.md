# llm-failover

Kill the worker mid-sentence. Another one finishes it, byte for byte.

When an LLM worker dies halfway through a generation, its KV cache dies with it. This project takes the position that the KV cache doesn't need to survive. Given the model weights, the token history, the sampler's RNG state and the current position, the next token is fully determined. So a replacement worker can replay the tokens, rebuild the cache and continue the exact same output. This repo builds that recovery path one piece at a time, using Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) `stories15M` model as the workload.

## Status

Done:

- **Engine** ([engine/](engine/)). llama2.c's forward pass, tokenizer and sampler wrapped in a C++ `Engine` class with `save()` and `restore()`. `restore()` rebuilds the KV cache by replaying the token history, then sets the RNG state. Output is byte-identical to llama2.c's own `run` for the same seed and prompt (checked by hand on 5 seed/prompt pairs, including one that hits the UTF-8 byte fallback).
- **Save/restore proof.** Generate 100 tokens straight, or generate 50, restore into a *different* engine and generate 50 more. Both give the same 100 tokens ([engine/test_engine.cpp](engine/test_engine.cpp)).
- **Worker** ([worker/worker.cpp](worker/worker.cpp)). TCP server that streams tokens as JSON lines. It answers `PING` and `STOP` between tokens, so it stays responsive mid-generation.
- **Cross-process resume.** Worker A streams 100 tokens. Worker B, a separate process, gets the first 50 via `RESUME` and produces the same last 50 tokens, positions and RNG states ([tests/test_worker.py](tests/test_worker.py)).
- **Router** ([router/router.py](router/router.py)). Python asyncio. Clients talk only to the router, which hands each session to a free worker and keeps the session log: epoch, position, token and RNG state for every token. A token is *committed* when it's in the log, and the client only ever sees committed tokens. A token is accepted only if its session, epoch and position are exactly what the log expects next. Duplicates, gaps, old epochs and sessions the worker doesn't own are dropped ([tests/test_router.py](tests/test_router.py)).
- **Client** ([client/client.py](client/client.py)). Sends a prompt to the router and prints the story as it streams.

Next, in order:

1. Heartbeats and recovery: `PING` every 200 ms, a worker is dead after 3 misses, and the router replays the log to another worker with the next epoch. Done when `kill -9` on a worker mid-story doesn't change the story. Right now a lost worker just fails the session, though the committed log survives.
2. CI test: SHA-256 of the failover run matches the uninterrupted run.
3. Measurements, including replay cost vs context length.

## Run it

Needs a C++17 compiler and Python 3.10+. No other dependencies.

```bash
make model     # downloads stories15M.bin (60 MB) and tokenizer.bin into models/
make test      # C++ engine tests + Python tests
```

Then, each in its own terminal:

```bash
make worker              # worker on 127.0.0.1:9001
make worker PORT=9002    # second worker
make router              # router on 127.0.0.1:9000, connected to both workers
python3 -m client.client --prompt "Once upon a time" --seed 42 --steps 100
```

```
Once upon a time, there was a boy named Timmy. Timmy loved to play with his toys. One day, he found a shiny thing on the ground. It was a ring with a pointy end. Timmy put it on his finger and said, "I think this is mine now."
Timmy's mom saw him and said, "Timmy, that ring is not yours. You should give it back." Timmy didn't want to give it
```

Same output as `./run stories15M.bin -s 42 -n 100 -i "Once upon a time"` in llama2.c.

## Protocol

One JSON object per line, over TCP.

Client and router:

| Direction | Message | Fields |
| --- | --- | --- |
| → router | `GENERATE` | `prompt?, seed?, steps?` |
| router → | `TEXT` | `pos, text` |
| router → | `END` | `session, reason` |
| router → | `ERROR` | `error` |

Router and worker. `steps` means the same as llama2.c's `-n`: the number of forward passes, prompt included (max 256).

| Direction | Message | Fields |
| --- | --- | --- |
| → worker | `START` | `session, epoch, prompt, seed, steps?` |
| → worker | `RESUME` | `session, epoch, tokens, rng_state, steps?` |
| → worker | `STOP` | `session` |
| → worker | `PING` | |
| worker → | `STARTED` | `session, epoch, tokens` (the prompt, encoded) |
| worker → | `RESUMED` | `session, epoch, pos, replay_ms` |
| worker → | `TOKEN` | `session, epoch, pos, token, rng_state` |
| worker → | `DONE` | `session, epoch, reason` (`steps` or `eos`) |
| worker → | `STOPPED` | `session` |
| worker → | `PONG` | |
| worker → | `ERROR` | `error` |

Every `TOKEN` carries the RNG state *after* that token was sampled. A log of `TOKEN`s is enough to resume on another worker: send the tokens so far plus the last `rng_state` in a `RESUME`.

## Design notes

- **Replay instead of shipping the KV cache.** For stories15M the full KV cache is about 3.5 MB (6 layers × 256 positions × 288 dims × K and V × 4 bytes). 256 tokens of history is about 1 KB. Replaying costs compute on recovery but keeps the durable state tiny.
- **Determinism has conditions.** Same weights, tokenizer, binary and CPU architecture on every worker. The matmul is single threaded so the float summation order never changes, and the build doesn't use `-ffast-math`, which would let the compiler reorder float math. This is not a claim of determinism across arbitrary hardware.
- **64-bit RNG state over JSON.** The worker's JSON reader keeps numbers as text, so `rng_state` round-trips all 64 bits. A double would silently drop the low bits past 2^53.
- **One session per worker, one thread.** Between tokens the worker polls its socket without blocking. That keeps `PING` latency under one token's time without threads or locks.
- **Log first, then the client.** Every token takes one extra hop through the router before the client sees it. That costs latency, but it means the client can never have seen a token that isn't in the log, so recovery can't produce a story that contradicts what the client already printed.
- **Epochs fence off old workers.** Each assignment of a session gets a new epoch, and anything sent under an old one is dropped. A worker that comes back after being declared dead can't write into a session that has moved on.
- **The router is a single point of failure.** Its log lives in memory. Worker failure is the focus here; replicating the router is out of scope.

## Credits

Model, tokenizer and the inference code in [engine/engine.cpp](engine/engine.cpp) come from Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) (MIT). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
