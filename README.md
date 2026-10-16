# llm-failover

Kill the worker mid-sentence. Another one finishes it, byte for byte.

When an LLM worker dies halfway through a generation, its KV cache dies with it. This project takes the position that the KV cache doesn't need to survive. Given the model weights, the token history, the sampler's RNG state and the current position, the next token is fully determined. So a replacement worker can replay the tokens, rebuild the cache and continue the exact same output. This repo builds that recovery path one piece at a time, using Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) `stories15M` model as the workload.

## Status

Done:

- **Engine** ([engine/](engine/)). llama2.c's forward pass, tokenizer and sampler wrapped in a C++ `Engine` class with `save()` and `restore()`. `restore()` rebuilds the KV cache by replaying the token history, then sets the RNG state. Output is byte-identical to llama2.c's own `run` for the same seed and prompt (checked by hand on 5 seed/prompt pairs, including one that hits the UTF-8 byte fallback).
- **Save/restore proof.** Generate 100 tokens straight, or generate 50, restore into a *different* engine and generate 50 more. Both give the same 100 tokens ([engine/test_engine.cpp](engine/test_engine.cpp)).
- **Worker** ([worker/worker.cpp](worker/worker.cpp)). TCP server that streams tokens as JSON lines. It answers `PING` and `STOP` between tokens, so it stays responsive mid-generation.
- **Cross-process resume.** Worker A streams 100 tokens. Worker B, a separate process, gets the first 50 via `RESUME` and produces the same last 50 tokens, positions and RNG states ([tests/test_worker.py](tests/test_worker.py)).
- **Client** ([client/client.py](client/client.py)). Sends a prompt and prints the story as it streams.

Next, in order:

1. Router keeps the session log (token, position, RNG state per step) with epochs. Client talks only to the router.
2. Heartbeats and failure detection: `kill -9` a worker mid-story and another one finishes it.
3. CI test: SHA-256 of the failover run matches the uninterrupted run.
4. Measurements, including replay cost vs context length.

## Run it

Needs a C++17 compiler and Python 3.10+. No other dependencies.

```bash
make model     # downloads stories15M.bin (60 MB) and tokenizer.bin into models/
make test      # C++ engine tests + Python tests
make worker    # worker on 127.0.0.1:9001
```

In another terminal:

```bash
python3 -m client.client --prompt "Once upon a time" --seed 42 --steps 100
```

```
Once upon a time, there was a boy named Timmy. Timmy loved to play with his toys. One day, he found a shiny thing on the ground. It was a ring with a pointy end. Timmy put it on his finger and said, "I think this is mine now."
Timmy's mom saw him and said, "Timmy, that ring is not yours. You should give it back." Timmy didn't want to give it
```

Same output as `./run stories15M.bin -s 42 -n 100 -i "Once upon a time"` in llama2.c.

## Worker protocol

One JSON object per line, over TCP. `steps` means the same as llama2.c's `-n`: the number of forward passes, prompt included (max 256).

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

`worker/server.py`, `worker/snapshot.py` and `router/server.py` are the first Python prototype: an HTTP snapshot store and proxy. The router in step 1 replaces them.

## Credits

Model, tokenizer and the inference code in [engine/engine.cpp](engine/engine.cpp) come from Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) (MIT). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
