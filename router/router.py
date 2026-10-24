"""Router: owns the session log and is the only thing clients talk to.

    python3 -m router.router --worker 127.0.0.1:9001 --worker 127.0.0.1:9002

A token is committed when the router appends it to the session log, and the client only
ever sees committed tokens. Every assignment of a session to a worker gets a new epoch.
A token is accepted only if its session, epoch and position are exactly what the log
expects next; everything else is dropped.

The router PINGs every worker every 200 ms. A worker that stays silent for 3 pings, or
whose connection drops, is dead. Its session is handed to another worker with the next
epoch, and that worker replays the log and continues from the last committed token.
"""

import argparse
import asyncio
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import TextIO

from router.tokenizer import decode, load_vocab

log = logging.getLogger("router")

RECONNECT_DELAY = 0.5
PING_INTERVAL = 0.2
MISSED_PINGS = 3
MAX_RECOVERIES = 3


@dataclass
class Record:
    epoch: int
    pos: int
    token: int
    rng_state: int  # after sampling this token, which is what a RESUME needs


@dataclass
class Session:
    id: str
    prompt: str
    seed: int
    steps: int | None
    out: asyncio.Queue = field(default_factory=asyncio.Queue)  # committed messages for the client
    state: str = "START"  # START -> RUNNING <-> RECOVERING, then DONE, STOPPED or FAILED
    epoch: int = 0
    worker: "Worker | None" = None
    tokens: list[int] = field(default_factory=list)  # prompt + every committed token
    log: list[Record] = field(default_factory=list)
    recoveries: int = 0
    lost_at: float = 0.0

    @property
    def rng_state(self) -> int:
        # After the last committed token; before any token is sampled it's still the seed
        return self.log[-1].rng_state if self.log else self.seed


class Worker:
    """One persistent connection to a C++ worker. Runs one session at a time."""

    def __init__(self, router: "Router", host: str, port: int):
        self.router = router
        self.host = host
        self.port = port
        self.writer: asyncio.StreamWriter | None = None
        self.session: Session | None = None
        # Only true once the worker has answered on this connection. A frozen process still
        # accepts TCP connections (the kernel does the handshake), so connecting proves nothing.
        self.ready = False
        self.last_seen = 0.0
        self.lost_reason = ""

    @property
    def name(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def free(self) -> bool:
        return self.ready and self.session is None

    def send(self, msg: dict):
        self.writer.write(json.dumps(msg).encode() + b"\n")

    async def run(self):
        while True:
            try:
                reader, self.writer = await asyncio.open_connection(self.host, self.port)
            except OSError:
                await asyncio.sleep(RECONNECT_DELAY)
                continue
            self.last_seen = time.monotonic()
            beat = asyncio.create_task(self.heartbeat())
            try:
                while line := await reader.readline():
                    self.last_seen = time.monotonic()
                    if not self.ready:
                        self.ready = True
                        log.info("worker %s ready", self.name)
                        self.router.event("worker_ready", worker=self.name)
                        self.router.worker_freed()
                    self.router.on_worker_message(self, json.loads(line))
            except (OSError, ValueError, KeyError, TypeError) as e:
                # A malformed message means something is badly wrong with that worker; reconnect
                log.warning("worker %s: %r", self.name, e)
                self.lost_reason = f"bad message: {e!r}"
            finally:
                beat.cancel()
                self.writer.close()
                self.writer = None
                was_ready, self.ready = self.ready, False
                reason, self.lost_reason = self.lost_reason or "connection closed", ""
                self.router.on_worker_lost(self, was_ready, reason)

    async def heartbeat(self):
        interval = self.router.ping_interval
        while True:
            silent = time.monotonic() - self.last_seen
            if silent > interval * self.router.missed_pings:
                self.lost_reason = f"silent for {silent * 1000:.0f} ms"
                if self.ready:
                    log.warning("worker %s %s, declaring it dead", self.name, self.lost_reason)
                # abort(), not close(): close() waits to flush, and a hung worker never reads
                self.writer.transport.abort()
                return
            self.send({"type": "PING"})
            await asyncio.sleep(interval)


class Router:
    def __init__(self, workers: list[tuple[str, int]], vocab: list[bytes], worker_timeout: float = 10.0,
                 ping_interval: float = PING_INTERVAL, missed_pings: int = MISSED_PINGS,
                 events: TextIO | None = None):
        self.vocab = vocab
        self.events = events
        self.workers = [Worker(self, host, port) for host, port in workers]
        self.worker_timeout = worker_timeout
        self.ping_interval = ping_interval
        self.missed_pings = missed_pings
        self.sessions: dict[str, Session] = {}
        self.dropped = 0
        self._freed = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()

    async def start(self, host: str, port: int) -> asyncio.Server:
        for w in self.workers:
            self.spawn(w.run())
        return await asyncio.start_server(self.handle_client, host, port)

    def event(self, name: str, **fields):
        """Append one JSON line to the event log: every commit, assignment, loss and recovery."""
        if self.events:
            self.events.write(json.dumps({"t": time.time(), "event": name, **fields}) + "\n")
            self.events.flush()

    def spawn(self, coro):
        # asyncio only keeps weak references to tasks, so hold on to them until they finish
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def close(self):
        tasks = list(self._tasks)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # Client side

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            try:
                session = self.new_session(json.loads(await reader.readline()))
            except (ValueError, AttributeError) as e:
                await self.write(writer, {"type": "ERROR", "error": f"bad request: {e}"})
                return
            try:
                worker = await asyncio.wait_for(self.acquire(), self.worker_timeout)
            except asyncio.TimeoutError:
                session.state = "FAILED"
                await self.write(writer, {"type": "ERROR", "error": "no worker available"})
                return
            self.assign(session, worker)
            await self.stream(session, reader, writer)
        except ConnectionError:
            pass
        finally:
            writer.close()

    def new_session(self, req: dict) -> Session:
        if req.get("type") != "GENERATE":
            raise ValueError("expected a GENERATE message")
        prompt = req.get("prompt", "")
        # A missing seed gets a random one, but it's recorded, so the run can still be replayed
        seed = req.get("seed", secrets.randbits(63) + 1)
        steps = req.get("steps")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if type(seed) is not int or not 0 < seed < 2**64:
            raise ValueError("seed must be an integer in [1, 2^64)")
        if steps is not None and (type(steps) is not int or steps < 1):
            raise ValueError("steps must be a positive integer")

        session = Session(id=secrets.token_hex(4), prompt=prompt, seed=seed, steps=steps)
        self.sessions[session.id] = session
        self.event("session", session=session.id, prompt=prompt, seed=seed, steps=steps)
        return session

    async def stream(self, session: Session, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        # Clients send nothing after GENERATE, so the read side finishing means they hung up
        hangup = asyncio.create_task(self.wait_eof(reader))
        try:
            while True:
                get = asyncio.create_task(session.out.get())
                done, _ = await asyncio.wait({get, hangup}, return_when=asyncio.FIRST_COMPLETED)
                if get not in done:
                    get.cancel()
                    self.stop(session)
                    return
                msg = get.result()
                await self.write(writer, msg)
                if msg["type"] in ("END", "ERROR"):
                    return
        except ConnectionError:
            self.stop(session)
        finally:
            hangup.cancel()

    @staticmethod
    async def wait_eof(reader: asyncio.StreamReader):
        while await reader.read(4096):
            pass

    @staticmethod
    async def write(writer: asyncio.StreamWriter, msg: dict):
        writer.write(json.dumps(msg).encode() + b"\n")
        await writer.drain()

    # Scheduling

    async def acquire(self) -> Worker:
        while True:
            for w in self.workers:
                if w.free:
                    return w
            self._freed.clear()
            await self._freed.wait()

    def worker_freed(self):
        self._freed.set()

    def assign(self, session: Session, worker: Worker):
        # New assignment, new epoch. Anything a worker sends under an older epoch is stale.
        session.epoch += 1
        session.state = "RUNNING"
        session.worker = worker
        worker.session = session
        msg = {"session": session.id, "epoch": session.epoch}
        if session.tokens:
            # Recovery. The log is the whole state: the worker replays it and carries on.
            msg |= {"type": "RESUME", "tokens": session.tokens, "rng_state": session.rng_state}
        else:
            msg |= {"type": "START", "prompt": session.prompt, "seed": session.seed}
        if session.steps is not None:
            msg["steps"] = session.steps
        worker.send(msg)
        log.info("session %s epoch %d -> worker %s (%s)", session.id, session.epoch, worker.name, msg["type"])
        self.event("assign", session=session.id, epoch=session.epoch, worker=worker.name,
                   kind=msg["type"], pos=len(session.tokens))

    def recover(self, session: Session, reason: str):
        session.worker = None
        if session.recoveries >= MAX_RECOVERIES:
            self.fail(session, f"{reason}, gave up after {MAX_RECOVERIES} recoveries")
            return
        session.recoveries += 1
        session.state = "RECOVERING"
        session.lost_at = time.monotonic()
        log.warning("session %s recovering at pos %d: %s", session.id, len(session.tokens), reason)
        self.event("recovering", session=session.id, pos=len(session.tokens), reason=reason,
                   attempt=session.recoveries)
        self.spawn(self.reassign(session))

    async def reassign(self, session: Session):
        try:
            worker = await asyncio.wait_for(self.acquire(), self.worker_timeout)
        except asyncio.TimeoutError:
            if session.state == "RECOVERING":
                self.fail(session, "no worker left to recover on")
            return
        # The client may have hung up while we waited
        if session.state == "RECOVERING":
            self.assign(session, worker)

    def stop(self, session: Session):
        if session.state == "RECOVERING":
            session.state = "STOPPED"
            self.event("stopped", session=session.id)
            return
        if session.state != "RUNNING":
            return
        session.state = "STOPPED"
        self.event("stopped", session=session.id)
        # The worker stays busy until it confirms with STOPPED
        session.worker.send({"type": "STOP", "session": session.id})
        log.info("session %s stopped: client hung up", session.id)

    def release(self, worker: Worker):
        worker.session = None
        self.worker_freed()

    def fail(self, session: Session, reason: str):
        session.state = "FAILED"
        session.out.put_nowait({"type": "ERROR", "error": reason})
        log.warning("session %s failed: %s", session.id, reason)
        self.event("failed", session=session.id, reason=reason)

    # Worker side

    def on_worker_message(self, worker: Worker, msg: dict):
        kind = msg.get("type")
        session = worker.session

        if kind == "PONG":
            return
        if kind == "STOPPED":
            if session and msg.get("session") == session.id:
                self.release(worker)
            return
        if kind == "ERROR":
            # Worker errors carry no session id; a worker runs one session, so it's this one
            if session:
                if session.state == "RUNNING":
                    self.fail(session, f"worker error: {msg.get('error')}")
                self.release(worker)
            return

        if session is None or msg.get("session") != session.id:
            self.drop(msg, f"session not assigned to {worker.name}")
            # A worker still working on a session it no longer owns is told to stop
            if msg.get("session"):
                worker.send({"type": "STOP", "session": msg["session"]})
            return
        if msg.get("epoch") != session.epoch:
            self.drop(msg, f"stale epoch, expected {session.epoch}")
            return
        if session.state != "RUNNING":
            self.drop(msg, f"session is {session.state}")
            return

        if kind == "STARTED":
            session.tokens = list(msg["tokens"])
            self.event("started", session=session.id, epoch=session.epoch, prompt_tokens=len(session.tokens))
            # Prompt tokens come from the client's own text, so they're committed as-is
            for pos in range(1, len(session.tokens)):
                self.emit_text(session, pos)
        elif kind == "RESUMED":
            since_lost = (time.monotonic() - session.lost_at) * 1000
            log.info("session %s resumed on %s at pos %d: replay %.0f ms, %.0f ms after the worker was lost",
                     session.id, worker.name, msg["pos"], msg["replay_ms"], since_lost)
            self.event("resumed", session=session.id, epoch=session.epoch, worker=worker.name, pos=msg["pos"],
                       replay_ms=msg["replay_ms"], since_lost_ms=round(since_lost, 2))
        elif kind == "TOKEN":
            self.commit(session, msg)
        elif kind == "DONE":
            session.state = "DONE"
            session.out.put_nowait({"type": "END", "session": session.id, "reason": msg.get("reason")})
            self.release(worker)
            log.info("session %s done: %d tokens", session.id, len(session.tokens))
            self.event("done", session=session.id, reason=msg.get("reason"), tokens=len(session.tokens))

    def commit(self, session: Session, msg: dict):
        pos = msg["pos"]
        if not session.tokens or pos != len(session.tokens):
            self.drop(msg, f"expected pos {len(session.tokens)}")
            return
        session.log.append(Record(msg["epoch"], pos, msg["token"], msg["rng_state"]))
        session.tokens.append(msg["token"])
        self.event("commit", session=session.id, epoch=msg["epoch"], pos=pos, token=msg["token"],
                   rng_state=msg["rng_state"])
        # Committed. Only now does the client get to see it.
        self.emit_text(session, pos)

    def emit_text(self, session: Session, pos: int):
        piece = decode(self.vocab, session.tokens[pos - 1], session.tokens[pos])
        if piece:
            session.out.put_nowait({"type": "TEXT", "pos": pos, "text": piece.decode("utf-8", errors="replace")})

    def drop(self, msg: dict, reason: str):
        self.dropped += 1
        log.warning("dropped %s (%s)", msg, reason)
        self.event("drop", reason=reason, msg=msg)

    def on_worker_lost(self, worker: Worker, was_ready: bool, reason: str):
        if was_ready:
            log.warning("worker %s lost: %s", worker.name, reason)
            self.event("worker_lost", worker=worker.name, reason=reason)
        session = worker.session
        worker.session = None
        if session and session.state == "RUNNING":
            self.recover(session, f"worker {worker.name} lost ({reason})")


def parse_addr(addr: str) -> tuple[str, int]:
    host, port = addr.rsplit(":", 1)
    return host, int(port)


async def serve(args):
    events = open(args.events, "w") if args.events else None
    router = Router([parse_addr(w) for w in args.worker], load_vocab(args.tokenizer), events=events)
    server = await router.start(args.host, args.port)
    host, port = server.sockets[0].getsockname()[:2]
    print(f"router listening on {host}:{port}", flush=True)
    async with server:
        await server.serve_forever()


def main():
    p = argparse.ArgumentParser(description="Route client sessions to workers and keep the token log.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--worker", action="append", required=True, help="host:port, repeat for more workers")
    p.add_argument("--tokenizer", default="models/tokenizer.bin")
    p.add_argument("--events", help="write a JSON line per router event (commits, failures, recoveries) here")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
