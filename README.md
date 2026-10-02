# llm-failover

Kill the worker mid-sentence. Another one finishes it, byte for byte. Fault tolerant LLM inference in C++.

## Day 1 milestone

This repo is being built in the order described in the design plan:

- verify deterministic generation with a fixed seed
- read and understand the llama2.c reference implementation
- create the repo skeleton and project notes
- keep the design doc local and out of Git history

## Current status

The reference model has been checked directly:

- same seed produces the same output
- different seed produces a different continuation

This proves the core Day 1 requirement before we add the custom engine or failover logic.

## Project structure

- engine/ - C++ runtime adapted from llama2.c
- worker/ - TCP worker process
- router/ - Python asyncio session router
- client/ - client for generating prompts
- tests/ - determinism and recovery checks
- docs/ - local design doc
- notes/ - daily working notes

## Read next

- [docs/DESIGN.md](docs/DESIGN.md)
- [notes/day1.md](notes/day1.md)

## Day 1 command check

From the repo root:

```bash
make day1
```
