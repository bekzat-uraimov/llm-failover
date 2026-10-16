import asyncio
import json
import logging
import os
import time
import unittest

from router.router import Router
from router.tokenizer import load_vocab
from tests.test_worker import MODEL, TOKENIZER, WORKER, start_worker

# Drops and disconnects are expected here; their warnings would bury the test output
logging.getLogger("router").setLevel(logging.CRITICAL)

HAVE_WORKER = os.path.exists(WORKER) and os.path.exists(MODEL)

# llama2.c: ./run stories15M.bin -s 42 -n 100 -i "Once upon a time", minus the trailing newline
STORY_42 = (
    'Once upon a time, there was a boy named Timmy. Timmy loved to play with his toys. One day, he found '
    'a shiny thing on the ground. It was a ring with a pointy end. Timmy put it on his finger and said, '
    '"I think this is mine now."\nTimmy\'s mom saw him and said, "Timmy, that ring is not yours. You should '
    'give it back." Timmy didn\'t want to give it'
)


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
        self.procs = []
        ports = []
        for _ in range(2):
            proc, port = start_worker()
            self.procs.append(proc)
            ports.append(port)
        await self.start_router([("127.0.0.1", p) for p in ports])
        await wait_until(lambda: all(w.free for w in self.router.workers))

    async def asyncTearDown(self):
        await super().asyncTearDown()
        for proc in self.procs:
            proc.kill()
            proc.wait()
            proc.stdout.close()

    async def test_story_through_router_matches_llama2c(self):
        msgs = await generate(self.port, {"prompt": "Once upon a time", "seed": 42, "steps": 100})

        self.assertEqual(text_of(msgs), STORY_42)
        self.assertEqual(msgs[-1]["type"], "END")
        self.assertEqual(msgs[-1]["reason"], "steps")

    async def test_log_holds_every_generated_token_once_in_order(self):
        msgs = await generate(self.port, {"prompt": "Once upon a time", "seed": 42, "steps": 100})

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
            generate(self.port, {"prompt": "Once upon a time", "seed": 42, "steps": 100}),
            generate(self.port, {"prompt": "Once upon a time", "seed": 42, "steps": 100}),
        )

        self.assertEqual(text_of(a), STORY_42)
        self.assertEqual(text_of(b), STORY_42)
        workers = {self.router.sessions[m[-1]["session"]].worker.name for m in (a, b)}
        self.assertEqual(len(workers), 2)

    async def test_more_clients_than_workers_wait_their_turn(self):
        runs = await asyncio.gather(*[
            generate(self.port, {"prompt": "Once upon a time", "seed": 42, "steps": 100}) for _ in range(4)
        ])
        self.assertEqual([text_of(r) for r in runs], [STORY_42] * 4)

    async def test_client_hangup_stops_the_worker(self):
        reader, writer = await connect(self.port, {"prompt": "Once upon a time", "seed": 42})
        await reader.readline()
        writer.close()
        await writer.wait_closed()

        (session,) = self.router.sessions.values()
        await wait_until(lambda: session.state == "STOPPED" and all(w.free for w in self.router.workers))
        msgs = await generate(self.port, {"prompt": "Once upon a time", "seed": 42, "steps": 100})
        self.assertEqual(text_of(msgs), STORY_42)

    async def test_bad_requests_get_an_error(self):
        for req in ["nope", "[1, 2]", json.dumps({"type": "HELLO"}),
                    {"seed": 0}, {"seed": 2**64}, {"seed": "42"}, {"steps": 0}, {"prompt": 5}]:
            msgs = await generate(self.port, req)
            self.assertEqual(len(msgs), 1, req)
            self.assertEqual(msgs[0]["type"], "ERROR", req)


class FakeWorker:
    """Plays the worker side so a test can send exactly the messages it wants."""

    async def start(self):
        self.received: asyncio.Queue = asyncio.Queue()
        self.connected = asyncio.Event()
        self.server = await asyncio.start_server(self.on_connect, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def on_connect(self, reader, writer):
        self.writer = writer
        self.connected.set()
        try:
            while line := await reader.readline():
                await self.received.put(json.loads(line))
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
        await self.start_router([("127.0.0.1", self.fake.port)])
        await wait_until(lambda: self.router.workers[0].free)
        vocab = load_vocab(TOKENIZER)
        self.once, self.upon, self.a, self.time = (vocab.index(w) for w in (b" Once", b" upon", b" a", b" time"))

    async def asyncTearDown(self):
        await super().asyncTearDown()
        await self.fake.close()

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

    async def test_worker_disconnect_fails_the_session(self):
        client, sid, epoch = await self.begin()
        self.fake.send(type="TOKEN", session=sid, epoch=epoch, pos=2, token=self.upon, rng_state=11)
        await wait_until(lambda: len(self.router.sessions[sid].log) == 1)

        self.fake.writer.close()
        msgs = await client

        self.assertEqual(msgs[-1]["type"], "ERROR")
        self.assertIn("disconnected", msgs[-1]["error"])
        self.assertEqual(self.router.sessions[sid].state, "FAILED")
        # What was committed before the crash stays in the log; that's what recovery will replay
        self.assertEqual(len(self.router.sessions[sid].log), 1)

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
