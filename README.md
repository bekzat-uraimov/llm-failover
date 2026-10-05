# llm-failover

Kill the worker mid-sentence. Another one finishes it, byte for byte.

This project explores a concrete systems problem in LLM serving: how do you continue a generation when a worker dies in the middle of a token stream without losing correctness or duplicating output? The goal is not to build a production inference stack; it is to study the design tradeoffs behind fault-tolerant, stateful inference using a tiny model as the workload.

## What this project is

At a high level, this repo models a minimal distributed setup with:

- a C++ inference engine adapted from llama2.c
- worker processes that generate text and hold ephemeral runtime state
- a Python router that manages sessions, determines when a worker has failed, and coordinates recovery
- a durable session log that stores enough information to rebuild generation on another worker

The core idea is simple but important:

- the KV cache and in-memory execution state are ephemeral
- the durable source of truth is the token history plus RNG state and position
- a replacement worker can rebuild the same generation by replaying the known state and continuing deterministically

This is a clean, small example of state recovery in a distributed AI system.

## Status: completed vs remaining

### Completed

- Deterministic generation was verified against the llama2.c reference implementation.
- Snapshot save/load support was implemented for token history, RNG state, and generation position.
- A worker-side snapshot API was added to persist resumable state.
- A router layer was added to proxy snapshot requests to the worker service.
- Integration tests were added for snapshot round-trips and router forwarding.
- The project structure was consolidated into a single repo rather than a day-by-day prototype.

### Remaining

- End-to-end failover orchestration across router + worker + client under real failure conditions
- Heartbeat and epoch-based worker recovery logic for session fencing and stale message rejection
- A more robust failover controller that handles retries, retries bounded by policy, and status transitions
- Production-quality validation, monitoring, and error handling for recovery scenarios
- Larger design work around model state persistence and recovery tradeoffs beyond the current minimal replay model

In short: the project has already proven the hard prerequisite pieces for deterministic recovery and state capture. The remaining work is the system-level orchestration that turns this into a full fault-tolerant execution path.

## Architecture

```text
client
  |
  v
router
  |
  +--> manages session state and recovery flow
  |
  +--> forwards snapshot/save and resume operations to workers
          |
          v
      worker(s)
          |
          +--> runs model inference
          +--> emits tokens and preserves recoverable state
```

## Repository layout

- engine/ — C++ runtime adapted from llama2.c
- worker/ — snapshot logic and HTTP worker service
- router/ — Python service that coordinates session-level recovery
- client/ — client-facing generation flow
- tests/ — deterministic generation, snapshot, and router validation
- docs/ — design notes and architecture background
- notes/ — historical day-by-day working notes retained for context

## Validation

The repo currently validates the core behavior with:

```bash
python3 -m unittest discover -v
```

The key behaviors under test include:

- seed determinism
- snapshot round-trip correctness
- resume-from-snapshot continuity
- router forwarding of snapshot operations

## Design intent

This repo is intentionally small and opinionated. It is designed to answer a systems question clearly:

> If the worker that is generating text disappears mid-stream, how can another worker continue without losing or duplicating output?

The answer here is not a production-grade serving platform. It is a focused study in deterministic replay, durable session state, and failover design decisions.

## Why this is a good engineering project

From a senior-engineering and hiring-manager perspective, this project demonstrates several strengths:

- strong systems thinking around failure modes and correctness
- practical understanding of state management in distributed computation
- experience working across C++ and Python components
- explicit attention to determinism, recovery, and reproducibility
- a clear project narrative with a defined milestone structure and remaining work

This is the kind of repo that reads like an engineering artifact rather than a loose collection of experiments: there is a concrete problem, a clear architecture, a bounded MVP, and a visible roadmap for what remains.

## Credits

This project builds on the work of Andrej Karpathy's llama2.c as a reference implementation for the tiny model runtime and generation loop.

## Notes

The early daily notes are preserved in the repo for historical context, but the project is now presented as a single coherent system design rather than a sequence of isolated day-specific snapshots.
