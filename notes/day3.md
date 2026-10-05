# Day 3 Notes

## Date

2026-10-04

## Goals for Day 3

- Add a minimal `router/` service to proxy client resume/save requests to `worker/` nodes.
- Create integration tests that start both services locally and verify proxying works end-to-end.
- Update Makefile with router serve target.

## What I implemented

- `router/server.py`: HTTP proxy that forwards `/snapshot/save` and `/snapshot/load` to worker at `http://localhost:8001`.
- `tests/test_router.py`: integration test that starts `worker.server` and `router.server` in background threads and verifies save/load via the router.

## Next steps

- Harden router: health checks, retries, worker discovery (DNS/consul), auth.
- Add router-level tests for failure modes (worker down / timeout).
- Wire router into higher-level client routing logic.
