import asyncio
import functools
import io
import json
import logging
import os
import signal
import subprocess
import time
import unittest

import router.router
from router.router import MAX_RECOVERIES, Router
from router.tokenizer import decode, load_vocab
from tests.test_worker import MODEL, ROOT, TOKENIZER, WORKER, Conn, start_worker

# Drops and disconnects are expected here; their warnings would bury the test output
logging.getLogger("router").setLevel(logging.CRITICAL)

HAVE_WORKER = os.path.exists(WORKER) and os.path.exists(MODEL)
# `make reference` builds llama2.c's own binary here
LLAMA2C = os.environ.get("LLAMA2C_RUN", os.path.join(ROOT, "build", "llama2.c", "run"))

STORY = {"prompt": "Once upon a time", "seed": 42, "steps": 100}


@functools.cache
def story() -> str:
    """STORY generated straight through one worker, no router, nothing killed: the reference run."""
    proc, port = start_worker()
    conn = Conn(port)
    try:
        conn.send({"type": "START", "session": "ref", "epoch": 1, **STORY})
        msgs = conn.recv_until("DONE")
    finally:
        conn.close()
        proc.kill()
        proc.wait()
        proc.stdout.close()
    tokens = msgs[0]["tokens"] + [m["token"] for m in msgs if m["type"] == "TOKEN"]
    vocab = load_vocab(TOKENIZER)
    return b"".join(decode(vocab, a, b) for a, b in zip(tokens, tokens[1:])).decode()


async def connect(port: int, req: dict | str):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    line = req if isinstance(req, str) else json.dumps({"type": "GENERATE", **req})
    writer.write(line.encode() + b"\n")
    await writer.drain()
    return reader, writer


async def generate(port: int, req: dict | str) -> list[dict]:
    reader, writer = await connect(port, req)
    msgs = []
    try:
        while line := await asyncio.wait_for(reader.readline(), 10):
            msgs.append(json.loads(line))
            if msgs[-1]["type"] in ("END", "ERROR"):
                break
    finally:
        writer.close()
        await writer.wait_closed()
    return msgs


def text_of(msgs: list[dict]) -> str:
    return "".join(m["text"] for m in msgs if m["type"] == "TEXT")


async def wait_until(cond, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


class RouterTestCase(unittest.IsolatedAsyncioTestCase):
    async def start_router(self, workers: list[tuple[str, int]], **kwargs):
        self.router = Router(workers, load_vocab(TOKENIZER), **kwargs)
        self.server = await self.router.start("127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        await self.router.close()


@unittest.skipUnless(HAVE_WORKER, "run `make` and `make model` first")
class RouterWithWorkersTest(RouterTestCase):
    async def asyncSetUp(self):
        # In a thread: blocking the event loop would look to the heartbeat like silent workers
        await asyncio.to_thread(story)
        self.procs = {}  # port -> worker process
        for _ in range(2):
            proc, port = start_worker()
            self.procs[port] = proc
        await self.start_router([("127.0.0.1", p) for p in self.procs])
        await wait_until(lambda: all(w.free for w in self.router.workers))

    async def asyncTearDown(self):
        await super().asyncTearDown()
        for proc in self.procs.values():
            proc.send_signal(signal.SIGCONT)
            proc.kill()
            proc.wait()
            proc.stdout.close()

    async def run_and_interrupt(self, sig: int, at_pos: int) -> tuple[list[dict], int]:
        """Stream STORY, and send `sig` to whichever worker is generating it at `at_pos`."""
        reader, writer = await connect(self.port, STORY)
        msgs, victim = [], None
        try:
            while line := await asyncio.wait_for(reader.readline(), 10):
                msgs.append(json.loads(line))
                if msgs[-1]["type"] in ("END", "ERROR"):
                    break
                if victim is None and msgs[-1].get("pos", 0) >= at_pos:
                    (session,) = self.router.sessions.values()
                    victim = session.worker.port
                    self.procs[victim].send_signal(sig)
        finally:
            writer.close()
            await writer.wait_closed()
        return msgs, victim

    async def test_kill_9_mid_story_gives_the_same_story(self):
        msgs, victim = await self.run_and_interrupt(signal.SIGKILL, at_pos=40)

        self.assertEqual(text_of(msgs), story())
        (session,) = self.router.sessions.values()
        self.assertEqual(session.state, "DONE")
        self.assertEqual(session.epoch, 2)
        self.assertNotEqual(session.worker.port, victim)
        # Tokens up to the crash came from epoch 1, the rest from epoch 2, no gaps, no repeats
        epochs = [r.epoch for r in session.log]
        self.assertEqual(epochs, sorted(epochs))
        self.assertEqual({1, 2}, set(epochs))
        self.assertEqual([r.pos for r in session.log], list(range(session.log[0].pos, 101)))

    async def test_frozen_worker_is_declared_dead_then_comes_back(self):
        msgs, victim = await self.run_and_interrupt(signal.SIGSTOP, at_pos=40)

        self.assertEqual(text_of(msgs), story())
        (session,) = self.router.sessions.values()
        self.assertEqual(session.epoch, 2)

        # Thawed, it notices the router dropped it, reconnects, and is usable again
        self.procs[victim].send_signal(signal.SIGCONT)
        await wait_until(lambda: all(w.free for w in self.router.workers))
        self.procs[session.worker.port].kill()
        again = await generate(self.port, STORY)
        self.assertEqual(text_of(again), story())

    async def test_story_through_router_matches_a_straight_run(self):
        msgs = await generate(self.port, STORY)

        self.assertEqual(text_of(msgs), story())
        self.assertEqual(msgs[-1]["type"], "END")
        self.assertEqual(msgs[-1]["reason"], "steps")

    @unittest.skipUnless(os.path.exists(LLAMA2C), "run `make reference` to build llama2.c")
    async def test_story_through_router_matches_llama2c_byte_for_byte(self):
        cases = [
            ("Once upon a time", 42, 100),
            ("Once upon a time", 43, 100),
            ("", 7, 256),
            ("The little dog", 1234, 200),
            ("Lily wanted a café ☕", 99, 150),  # ☕ isn't in the vocab: byte fallback
        ]
        for prompt, seed, steps in cases:
            msgs = await generate(self.port, {"prompt": prompt, "seed": seed, "steps": steps})
            # Async, so the router's heartbeat keeps running while llama2.c works
            proc = await asyncio.create_subprocess_exec(
                LLAMA2C, MODEL, "-z", TOKENIZER, "-s", str(seed), "-n", str(steps), "-i", prompt,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            ref, _ = await proc.communicate()
            # llama2.c ends its output with a newline
            self.assertEqual(text_of(msgs).encode() + b"\n", ref, (prompt, seed))

    async def test_log_holds_every_generated_token_once_in_order(self):
        msgs = await generate(self.port, STORY)

        session = self.router.sessions[msgs[-1]["session"]]
        n_prompt = len(session.tokens) - len(session.log)
        self.assertEqual(session.state, "DONE")
        self.assertEqual(len(session.tokens), 101)
        self.assertEqual([r.pos for r in session.log], list(range(n_prompt, 101)))
        self.assertEqual([r.token for r in session.log], session.tokens[n_prompt:])
        self.assertTrue(all(r.epoch == 1 for r in session.log))
        self.assertEqual(self.router.dropped, 0)

    async def test_client_only_sees_committed_tokens(self):
        reader, writer = await connect(self.port, {"prompt": "Once upon a time", "seed": 42, "steps": 60})
        try:
            while msg := json.loads(await reader.readline()):
                if msg["type"] == "END":
                    break
                (session,) = self.router.sessions.values()
                # At the moment the client reads position p, the log already has it
                self.assertLess(msg["pos"], len(session.tokens))
        finally:
            writer.close()
            await writer.wait_closed()

    async def test_two_clients_run_at_once_on_different_workers(self):
        a, b = await asyncio.gather(
            generate(self.port, STORY),
            generate(self.port, STORY),
        )

        self.assertEqual(text_of(a), story())
        self.assertEqual(text_of(b), story())
        workers = {self.router.sessions[m[-1]["session"]].worker.name for m in (a, b)}
        self.assertEqual(len(workers), 2)

    async def test_more_clients_than_workers_wait_their_turn(self):
        runs = await asyncio.gather(*[
            generate(self.port, STORY) for _ in range(4)
        ])
        self.assertEqual([text_of(r) for r in runs], [story()] * 4)

    async def test_client_hangup_stops_the_worker(self):
        reader, writer = await connect(self.port, {"prompt": "Once upon a time", "seed": 42})
        await reader.readline()
        writer.close()
        await writer.wait_closed()

        (session,) = self.router.sessions.values()
        await wait_until(lambda: session.state == "STOPPED" and all(w.free for w in self.router.workers))
        msgs = await generate(self.port, STORY)
        self.assertEqual(text_of(msgs), story())

    async def test_bad_requests_get_an_error(self):
        for req in ["nope", "[1, 2]", json.dumps({"type": "HELLO"}),
                    {"seed": 0}, {"seed": 2**64}, {"seed": "42"}, {"steps": 0}, {"prompt": 5}]:
            msgs = await generate(self.port, req)
            self.assertEqual(len(msgs), 1, req)
            self.assertEqual(msgs[0]["type"], "ERROR", req)


class FakeWorker:
    """Plays the worker side so a test can send exactly the messages it wants.

    Answers PINGs by itself unless frozen; a new connection thaws it, like a restarted worker.
    """

    async def start(self):
        self.received: asyncio.Queue = asyncio.Queue()
        self.frozen = False
        self.server = await asyncio.start_server(self.on_connect, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def on_connect(self, reader, writer):
        self.writer = writer
        self.frozen = False
        try:
            while line := await reader.readline():
                msg = json.loads(line)
                if msg["type"] != "PING":
                    await self.received.put(msg)
                elif not self.frozen:
                    writer.write(b'{"type": "PONG"}\n')
        finally:
            writer.close()

    def send(self, **msg):
        self.writer.write(json.dumps(msg).encode() + b"\n")

    async def recv(self) -> dict:
        return await asyncio.wait_for(self.received.get(), 5)

    async def close(self):
        self.server.close()
        self.writer.close()
        await self.server.wait_closed()


class RouterFencingTest(RouterTestCase):
    async def asyncSetUp(self):
        self.fake = FakeWorker()
        await self.fake.start()
        self.reconnect_delay = router.router.RECONNECT_DELAY
        router.router.RECONNECT_DELAY = 0.05
        # Dead after 3 x 50 ms of silence, to keep the tests quick
        await self.start_router([("127.0.0.1", self.fake.port)], ping_interval=0.05)
        await wait_until(lambda: self.router.workers[0].free)
        vocab = load_vocab(TOKENIZER)
        self.once, self.upon, self.a, self.time = (vocab.index(w) for w in (b" Once", b" upon", b" a", b" time"))

    async def asyncTearDown(self):
        await super().asyncTearDown()
        await self.fake.close()
        router.router.RECONNECT_DELAY = self.reconnect_delay

    async def begin(self) -> tuple[asyncio.Task, str, int]:
        client = asyncio.create_task(generate(self.port, {"prompt": "Once", "seed": 5}))
        start = await self.fake.recv()
        self.assertEqual(start["type"], "START")
        self.fake.send(type="STARTED", session=start["session"], epoch=start["epoch"], tokens=[1, self.once])
        return client, start["session"], start["epoch"]

    async def test_only_the_expected_next_token_is_committed(self):
        client, sid, epoch = await self.begin()
        tok = dict(type="TOKEN", session=sid, epoch=epoch)

        self.fake.send(**tok, pos=2, token=self.upon, rng_state=11)
        self.fake.send(**tok, pos=2, token=self.time, rng_state=12)                # duplicate position
        self.fake.send(**tok, pos=4, token=self.time, rng_state=13)                # gap
        self.fake.send(**{**tok, "epoch": epoch - 1}, pos=3, token=self.time, rng_state=14)  # old epoch
        self.fake.send(**tok, pos=3, token=self.a, rng_state=15)
        self.fake.send(type="DONE", session=sid, epoch=epoch, reason="steps")
        msgs = await client

        session = self.router.sessions[sid]
        self.assertEqual([(r.pos, r.token, r.rng_state) for r in session.log], [(2, self.upon, 11), (3, self.a, 15)])
        self.assertEqual(session.tokens, [1, self.once, self.upon, self.a])
        self.assertEqual(self.router.dropped, 3)
        self.assertEqual(text_of(msgs), "Once upon a")
        self.assertEqual([m["pos"] for m in msgs if m["type"] == "TEXT"], [1, 2, 3])

    async def test_worker_is_told_to_stop_a_session_it_does_not_own(self):
        client, sid, epoch = await self.begin()

        self.fake.send(type="TOKEN", session="someone-else", epoch=1, pos=2, token=self.upon, rng_state=1)

        self.assertEqual(await self.fake.recv(), {"type": "STOP", "session": "someone-else"})
        self.assertEqual(self.router.dropped, 1)
        self.fake.send(type="DONE", session=sid, epoch=epoch, reason="steps")
        await client

    async def test_tokens_after_done_are_dropped(self):
        client, sid, epoch = await self.begin()
        self.fake.send(type="DONE", session=sid, epoch=epoch, reason="eos")
        await client

        self.fake.send(type="TOKEN", session=sid, epoch=epoch, pos=2, token=self.upon, rng_state=1)

        await wait_until(lambda: self.router.dropped == 1)
        self.assertEqual(self.router.sessions[sid].tokens, [1, self.once])

    async def test_lost_worker_gets_a_resume_built_from_the_log(self):
        client, sid, epoch = await self.begin()
        self.fake.send(type="TOKEN", session=sid, epoch=epoch, pos=2, token=self.upon, rng_state=11)
        await wait_until(lambda: len(self.router.sessions[sid].log) == 1)

        self.fake.writer.close()
        resume = await self.fake.recv()

        self.assertEqual(resume, {"type": "RESUME", "session": sid, "epoch": epoch + 1,
                                  "tokens": [1, self.once, self.upon], "rng_state": 11})
        self.fake.send(type="RESUMED", session=sid, epoch=epoch + 1, pos=3, replay_ms=1.0)
        self.fake.send(type="TOKEN", session=sid, epoch=epoch + 1, pos=3, token=self.a, rng_state=12)
        self.fake.send(type="DONE", session=sid, epoch=epoch + 1, reason="steps")
        msgs = await client
        self.assertEqual(text_of(msgs), "Once upon a")
        self.assertEqual([r.epoch for r in self.router.sessions[sid].log], [1, 2])

    async def test_event_log_records_the_failover(self):
        self.router.events = io.StringIO()
        client, sid, epoch = await self.begin()
        self.fake.send(type="TOKEN", session=sid, epoch=epoch, pos=2, token=self.upon, rng_state=11)
        await wait_until(lambda: len(self.router.sessions[sid].log) == 1)
        self.fake.writer.close()
        await self.fake.recv()
        self.fake.send(type="RESUMED", session=sid, epoch=epoch + 1, pos=3, replay_ms=1.5)
        self.fake.send(type="DONE", session=sid, epoch=epoch + 1, reason="steps")
        await client

        events = [json.loads(line) for line in self.router.events.getvalue().splitlines()]
        names = [e["event"] for e in events if e["event"] != "worker_ready"]
        self.assertEqual(names, ["session", "assign", "started", "commit", "worker_lost",
                                 "recovering", "assign", "resumed", "done"])
        by_name = {e["event"]: e for e in events}
        self.assertEqual(by_name["commit"]["rng_state"], 11)
        self.assertEqual(by_name["recovering"]["pos"], 3)
        self.assertEqual(by_name["resumed"]["replay_ms"], 1.5)
        self.assertTrue(all(isinstance(e["t"], float) for e in events))

    async def test_silent_worker_is_declared_dead_and_its_old_epoch_is_fenced_off(self):
        client, sid, epoch = await self.begin()
        self.fake.frozen = True

        # 3 missed pings later the router hangs up on it and, once it's back, resumes there
        resume = await self.fake.recv()
        self.assertEqual((resume["type"], resume["epoch"]), ("RESUME", epoch + 1))

        # A token the worker generated under the old assignment arrives late: dropped
        self.fake.send(type="TOKEN", session=sid, epoch=epoch, pos=2, token=self.time, rng_state=99)
        self.fake.send(type="TOKEN", session=sid, epoch=epoch + 1, pos=2, token=self.upon, rng_state=11)
        self.fake.send(type="DONE", session=sid, epoch=epoch + 1, reason="steps")
        msgs = await client

        self.assertEqual(text_of(msgs), "Once upon")
        self.assertEqual(self.router.dropped, 1)

    async def test_lost_before_started_starts_over(self):
        client = asyncio.create_task(generate(self.port, {"prompt": "Once", "seed": 5}))
        first = await self.fake.recv()
        self.fake.writer.close()

        second = await self.fake.recv()

        # Nothing was committed, so there is nothing to replay: same START, next epoch
        self.assertEqual(second, {**first, "epoch": first["epoch"] + 1})
        self.fake.send(type="STARTED", session=second["session"], epoch=second["epoch"], tokens=[1, self.once])
        self.fake.send(type="DONE", session=second["session"], epoch=second["epoch"], reason="eos")
        self.assertEqual(text_of(await client), "Once")

    async def test_session_fails_after_max_recoveries(self):
        client, sid, epoch = await self.begin()
        for _ in range(MAX_RECOVERIES):
            self.fake.writer.close()
            self.assertEqual((await self.fake.recv())["type"], "RESUME")
        self.fake.writer.close()

        msgs = await client
        self.assertEqual(msgs[-1]["type"], "ERROR")
        self.assertIn("gave up", msgs[-1]["error"])
        self.assertEqual(self.router.sessions[sid].state, "FAILED")
        self.assertEqual(self.router.sessions[sid].epoch, MAX_RECOVERIES + 1)

    async def test_worker_error_is_passed_to_the_client(self):
        client, sid, epoch = await self.begin()
        self.fake.send(type="ERROR", error="prompt too long")
        msgs = await client

        self.assertEqual(msgs[-1], {"type": "ERROR", "error": "worker error: prompt too long"})
        await wait_until(lambda: self.router.workers[0].free)


class RouterNoWorkerTest(RouterTestCase):
    async def test_no_worker_available_is_an_error(self):
        await self.start_router([], worker_timeout=0.1)
        msgs = await generate(self.port, {"prompt": "Once", "seed": 5})
        self.assertEqual(msgs, [{"type": "ERROR", "error": "no worker available"}])


if __name__ == "__main__":
    unittest.main()
