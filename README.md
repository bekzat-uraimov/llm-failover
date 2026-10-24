# llm-failover

[![ci](https://github.com/bekzat-uraimov/llm-failover/actions/workflows/ci.yml/badge.svg)](https://github.com/bekzat-uraimov/llm-failover/actions/workflows/ci.yml)

Kill the worker mid-sentence. Another one finishes it, byte for byte.

![Demo: the worker is killed at token 80, another worker finishes the story, and both runs have the same SHA-256](assets/demo.gif)

When an LLM worker dies halfway through a generation, its KV cache dies with it. This project takes the position that the KV cache doesn't need to survive. Given the model weights, the token history, the sampler's RNG state and the current position, the next token is fully determined. So a replacement worker can replay the tokens, rebuild the cache and continue the exact same output.

It's a small study of one systems problem, not a serving stack: C++ workers running Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) `stories15M` model, a Python router that keeps the token log, and a test that kills workers mid-story and compares SHA-256 against an uninterrupted run.

## How it works

```
             +---------+
  client --> | router  |  session log: (session, epoch, pos, token, rng_state)
  (prompt)   |         |  the client only sees a token once it's logged
             +----+----+
                  |  PING every 200 ms, START / RESUME, TOKEN stream
          +-------+-------+
          |               |
     +----v----+     +----v----+
     | worker A|     | worker B|   each runs the whole model, one session at a time
     +---------+     +---------+
```

1. The router assigns a session to a free worker with epoch 1. The worker streams `TOKEN {pos, token, rng_state}`, where `rng_state` is the sampler state *after* that token.
2. The router logs each token, then forwards its text to the client. Tokens with the wrong session, epoch or position are dropped.
3. Worker A dies: its socket closes, or it misses 3 heartbeats (about 600 ms).
4. The router sends worker B a `RESUME` with epoch 2: every logged token plus the last `rng_state`. B replays the tokens through the model to rebuild the KV cache, sets the RNG, and keeps going. The next token is the one A would have produced.

## What's verified

| Claim | Where |
| --- | --- |
| Same seed gives the same tokens; restoring into a fresh engine continues identically | [engine/test_engine.cpp](engine/test_engine.cpp) |
| Output is byte-identical to llama2.c's own `run`, built from source on the same machine, on 5 seed/prompt pairs including a UTF-8 byte-fallback prompt | [tests/test_router.py](tests/test_router.py) |
| A second worker process given the first 50 tokens produces the same last 50 tokens, positions and RNG states | [tests/test_worker.py](tests/test_worker.py) |
| Duplicates, gaps, old epochs and sessions a worker doesn't own are dropped | [tests/test_router.py](tests/test_router.py) |
| `kill -9` or `kill -STOP` on the generating worker at tokens 10, 80 and 150 gives the same SHA-256 as the normal run | [scripts/demo.py](scripts/demo.py) `--check` |

CI runs all of it on Ubuntu (gcc, Python 3.10) and macOS on Apple Silicon (clang, Python 3.13).

Not done: running the workers on two separate machines. Everything above is processes on one host.

## Run it

Needs a C++17 compiler and Python 3.10+. No other dependencies.

```bash
make model     # downloads stories15M.bin (60 MB) and tokenizer.bin, checks their SHA-256
make test      # builds everything, plus llama2.c from source as the reference
make demo      # the GIF above: normal run, then kill -9 at token 80, compare SHA-256
```

`python3 scripts/demo.py --signal stop` freezes the worker instead of killing it, so only the heartbeat can notice.

To run the pieces by hand, each in its own terminal:

```bash
make worker              # worker on 127.0.0.1:9001
make worker PORT=9002    # second worker
make router              # router on 127.0.0.1:9000, connected to both
python3 -m client.client --prompt "Once upon a time" --seed 42 --steps 100
```

Kill either worker while the story streams; the client just sees a short pause. `python3 -m router.router --events events.jsonl ...` writes one JSON line per commit, assignment, loss and recovery.

## Measurements

Measured on Apple M3 Pro, macOS, Apple clang version 16.0.0 (clang-1600.0.26.6), Python 3.14.5. Median of 3 runs. Story: seed 42, "Once upon a time".

| One session, through the router | |
| --- | --- |
| tokens per second | 155 |
| time between tokens, median | 6.4 ms |
| time between tokens, p99 | 7.1 ms |

Horizontal scaling here means serving more sessions at once, not making one story faster. Every token of a session comes from one worker:

| Sessions at once | 1 worker | 2 workers |
| --- | --- | --- |
| 1 | 151 tok/s | 139 tok/s |
| 2 | 148 tok/s | 281 tok/s |
| 4 | 135 tok/s | 280 tok/s |

One session runs on one worker either way; the 151 vs 139 difference is noise.

Failover pause is the gap the client sees between the last token before the interruption and the first one after. After `kill -9` it's almost all replay. After `kill -STOP` add the heartbeat timeout, about 700 ms:

| Worker interrupted at token | `kill -9`: pause seen by client | `kill -STOP`: pause seen by client |
| --- | --- | --- |
| 64 | 422 ms | 1172 ms |
| 128 | 863 ms | 1490 ms |
| 192 | 1323 ms | 2121 ms |

Replay time grows with context length, slightly faster than linearly, because attention at position *n* looks at all *n* earlier positions:

| Context length | Replay time | Per token | RESUME message |
| --- | --- | --- | --- |
| 64 tokens | 427 ms | 6.77 ms | 477 bytes |
| 128 tokens | 876 ms | 6.89 ms | 844 bytes |
| 256 tokens | 1882 ms | 7.38 ms | 1,594 bytes |

| State for a full 256-token context | Size |
| --- | --- |
| KV cache (6 layers × 256 positions × 288 dims × K and V × 4 bytes) | 3.5 MB |
| Router log (token id + RNG state per token) | 3.1 KB |
| What a RESUME needs (tokens + the last RNG state) | 1.0 KB |

Reproduce with `python3 scripts/measure.py`.

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

## Design notes

- **Replay instead of shipping the KV cache.** A full context's KV cache is 3.5 MB; the tokens plus the last RNG state are 1 KB. Replay costs compute on recovery, about 7 ms per token here, so 1.9 s for a full context. For a tiny model that's the right trade. For a large model with long contexts, copying the cache would start to win.
- **Determinism has conditions.** Same weights, tokenizer, binary and CPU architecture on every worker. The matmul is single threaded so the float summation order never changes, and the build doesn't use `-ffast-math`, which would let the compiler reorder float math. This is not a claim of determinism across arbitrary hardware, which is why the tests compare against llama2.c built on the same machine rather than against a saved output.
- **Log first, then the client.** Every token takes one extra hop through the router before the client sees it. That costs latency, but the client can never have seen a token that isn't in the log, so a recovery can't contradict what the client already printed.
- **Epochs fence off old workers.** Each assignment of a session gets a new epoch, and anything sent under an old one is dropped. A worker that comes back after being declared dead can't write into a session that has moved on.
- **Heartbeat timeout is a tradeoff.** 3 missed pings at 200 ms means a frozen worker costs well under a second of detection. Shorter would recover faster but would also kill workers that are only slow for a moment.
- **Replay doesn't block the heartbeat.** A full 256-token replay takes longer than the 600 ms a worker may stay silent. So the worker replays one position per loop and answers `PING`s in between, the same way it does while generating. Otherwise the heartbeat would kill the worker doing the recovery.
- **Connected doesn't mean alive.** The kernel completes a TCP handshake even for a frozen process. After reconnecting, a worker only gets sessions once it has answered a message.
- **64-bit RNG state over JSON.** The worker's JSON reader keeps numbers as text, so `rng_state` round-trips all 64 bits. A double would silently drop the low bits past 2^53.
- **One session per worker, one thread.** No locks, and `PING` latency stays under one token's time. More sessions at once means more workers, not more threads.
- **The router is a single point of failure.** Its log lives in memory. Worker failure is the focus here; replicating the router is out of scope.

## Credits

Model, tokenizer and the inference code in [engine/engine.cpp](engine/engine.cpp) come from Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) (MIT). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
