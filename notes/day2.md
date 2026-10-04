# Day 2 Notes

## Date

2026-10-03

## Goals for Day 2

- Define a compact durable snapshot format for failover: token history + RNG state + current position.
- Design minimal recovery API to load a snapshot and resume generation deterministically.
- Implement snapshot save/load in the worker and wire through the router.
- Add tests to prove deterministic recovery across process restart and node failover.

## Snapshot format (proposal)

- token_history: array of token IDs (small, incremental growth)
- rng_state: serialized PRNG state (pickle/byte array) — must be exact for deterministic sampling
- position: integer (current token position)
- metadata: model name, tokenizer id, seed, timestamp, optional KV-checksum

Format notes:
- store as a small binary or JSON+base64 for portability.
- include a KV-checksum field (optional) so the verifier can detect mismatch if using cached KV.

## Recovery API (minimal)

- POST /snapshot/save -> saves current snapshot (returns snapshot id)
- POST /snapshot/load -> loads snapshot id and resumes generation
- GET /snapshot/:id -> returns metadata for snapshot

Router/worker responsibilities:

- Worker: produce snapshots on-demand or periodically, ensure RNG state and token_history are recorded atomically.
- Router: accept client resume requests, forward snapshots to worker, and coordinate resumption.

## Tests to add

- Snapshot round-trip: save snapshot, load into new process, generate N tokens — outputs must match original.
- Seed divergence: different seed produces different output.
- Partial-history resume: resume from token 50 and continue 50 tokens; compare with original run.

## Repro commands / examples

- Run original: `./run stories15M.bin -s 42 -n 100` (record token_history + rng_state at position 50)
- Save snapshot via API: `curl -X POST http://localhost:PORT/snapshot/save` (worker should return id)
- Load snapshot: `curl -X POST http://localhost:PORT/snapshot/load -d '{"id":"..."}'`

## Next steps (short-term)

1. Implement snapshot serialization in `worker/` (save/load functions).
2. Add router endpoints and simple CLI to trigger snapshot save/load.
3. Write tests in `tests/` to validate deterministic recovery.
4. Run experiments and record results back into these notes.

## Decisions / Assumptions

- RNG state must be fully serializable and restored exactly to guarantee reproducibility.
- KV cache will not be transported initially; instead we will rebuild KV by replaying `token_history` on resume. This keeps snapshots small.

## Open questions

- Should snapshots include a small compressed KV to speed resume for large models?
- What retention/garbage collection policy for snapshots makes sense in production?

---

Created as the Day 2 planning and specification draft. Implementations follow next.

## Experiment results (resume roundtrip)

- Test: generate 100 tokens with seed 42; save snapshot at token 50; reload snapshot and resume 50 tokens.
- Result: resumed sequence matched the original continuation exactly (test `tests/test_resume_experiment.py` passed).

