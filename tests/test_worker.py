import json
import os
import socket
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "build", "worker")
MODEL = os.path.join(ROOT, "models", "stories15M.bin")
TOKENIZER = os.path.join(ROOT, "models", "tokenizer.bin")
PROMPT = "Once upon a time"


def start_worker() -> tuple[subprocess.Popen, int]:
    proc = subprocess.Popen(
        [WORKER, "--model", MODEL, "--tokenizer", TOKENIZER, "--port", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    # First stdout line is "worker listening on 127.0.0.1:<port>"
    port = int(proc.stdout.readline().rsplit(":", 1)[1])
    return proc, port


class Conn:
    def __init__(self, port: int):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.f = self.sock.makefile("rwb")

    def send(self, msg: dict | str):
        line = msg if isinstance(msg, str) else json.dumps(msg)
        self.f.write(line.encode() + b"\n")
        self.f.flush()

    def recv(self) -> dict:
        line = self.f.readline()
        if not line:
            raise ConnectionError("worker closed the connection")
        return json.loads(line)

    def recv_until(self, *types: str) -> list[dict]:
        msgs = []
        while True:
            msg = self.recv()
            msgs.append(msg)
            if msg["type"] in types:
                return msgs

    def close(self):
        self.f.close()
        self.sock.close()


def tokens_of(msgs: list[dict]) -> list[dict]:
    return [m for m in msgs if m["type"] == "TOKEN"]


@unittest.skipUnless(os.path.exists(WORKER) and os.path.exists(MODEL), "run `make` and `make model` first")
class WorkerTest(unittest.TestCase):
    def setUp(self):
        self.proc, self.port = start_worker()
        self.conn = Conn(self.port)

    def tearDown(self):
        self.conn.close()
        self.proc.kill()
        self.proc.wait()
        self.proc.stdout.close()

    def generate(self, conn: Conn, seed: int, steps: int) -> list[dict]:
        conn.send({"type": "START", "session": "s", "epoch": 1, "prompt": PROMPT, "seed": seed, "steps": steps})
        return conn.recv_until("DONE", "ERROR")

    def test_start_streams_tokens_in_order_then_done(self):
        msgs = self.generate(self.conn, seed=42, steps=40)

        started, done = msgs[0], msgs[-1]
        self.assertEqual(started["type"], "STARTED")
        self.assertEqual(started["tokens"][0], 1)
        self.assertEqual(done, {"type": "DONE", "session": "s", "epoch": 1, "reason": "steps"})
        positions = [m["pos"] for m in tokens_of(msgs)]
        first = len(started["tokens"])
        self.assertEqual(positions, list(range(first, 41)))

    def test_same_seed_gives_same_tokens(self):
        a = [m["token"] for m in tokens_of(self.generate(self.conn, seed=42, steps=60))]
        b = [m["token"] for m in tokens_of(self.generate(self.conn, seed=42, steps=60))]
        c = [m["token"] for m in tokens_of(self.generate(self.conn, seed=43, steps=60))]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_resume_on_another_worker_finishes_the_same_story(self):
        msgs = self.generate(self.conn, seed=42, steps=100)
        prompt = msgs[0]["tokens"]
        streamed = tokens_of(msgs)

        # Pretend this worker died after token 50: hand the log so far to a second worker
        cut = 50 - len(prompt)
        tokens = prompt + [m["token"] for m in streamed[:cut]]
        other, port = start_worker()
        conn = Conn(port)
        try:
            conn.send({"type": "RESUME", "session": "s", "epoch": 2, "tokens": tokens,
                       "rng_state": streamed[cut - 1]["rng_state"], "steps": 100})
            resumed = conn.recv_until("DONE", "ERROR")
        finally:
            conn.close()
            other.kill()
            other.wait()
            other.stdout.close()

        self.assertEqual(resumed[0]["type"], "RESUMED")
        self.assertEqual(resumed[0]["pos"], 50)
        rest = [(m["pos"], m["token"], m["rng_state"]) for m in tokens_of(resumed)]
        expected = [(m["pos"], m["token"], m["rng_state"]) for m in streamed[cut:]]
        self.assertEqual(rest, expected)
        self.assertTrue(all(m["epoch"] == 2 for m in tokens_of(resumed)))

    def test_ping_is_answered_mid_generation(self):
        self.conn.send({"type": "START", "session": "s", "epoch": 1, "prompt": PROMPT, "seed": 42})
        self.conn.send({"type": "PING"})
        msgs = self.conn.recv_until("PONG")

        self.assertNotIn("DONE", [m["type"] for m in msgs])
        self.conn.send({"type": "STOP", "session": "s"})
        self.conn.recv_until("STOPPED")

    def test_stop_ends_the_stream(self):
        self.conn.send({"type": "START", "session": "s", "epoch": 1, "prompt": PROMPT, "seed": 42})
        self.conn.recv_until("TOKEN")

        self.conn.send({"type": "STOP", "session": "s"})
        msgs = self.conn.recv_until("STOPPED")
        self.conn.send({"type": "PING"})

        self.assertEqual(msgs[-1], {"type": "STOPPED", "session": "s"})
        self.assertEqual(self.conn.recv(), {"type": "PONG"})

    def test_start_while_busy_is_an_error(self):
        self.conn.send({"type": "START", "session": "a", "epoch": 1, "prompt": PROMPT, "seed": 42})
        self.conn.send({"type": "START", "session": "b", "epoch": 1, "prompt": PROMPT, "seed": 42})
        msgs = self.conn.recv_until("ERROR")

        self.assertIn("busy", msgs[-1]["error"])
        self.conn.send({"type": "STOP", "session": "a"})
        self.conn.recv_until("STOPPED")

    def test_bad_messages_get_an_error_and_keep_the_connection(self):
        bad = [
            "not json",
            {"type": "LAUNCH"},
            {"type": "START", "session": "s", "epoch": 1, "prompt": PROMPT, "seed": 0},
            {"type": "START", "session": "s", "epoch": 1, "prompt": PROMPT, "seed": -5},
            {"type": "RESUME", "session": "s", "epoch": 1, "tokens": [5, 6], "rng_state": 1},
            {"type": "RESUME", "session": "s", "epoch": 1, "tokens": [1, 999999], "rng_state": 1},
            {"type": "RESUME", "session": "s", "epoch": 1, "tokens": [1, 6], "rng_state": 1, "pos": 9},
        ]
        for msg in bad:
            self.conn.send(msg)
            self.assertEqual(self.conn.recv()["type"], "ERROR", msg)

        self.conn.send({"type": "PING"})
        self.assertEqual(self.conn.recv(), {"type": "PONG"})

    def test_rng_state_keeps_all_64_bits(self):
        msgs = self.generate(self.conn, seed=2**64 - 1, steps=10)
        self.assertEqual(msgs[0]["type"], "STARTED")
        self.assertTrue(any(m["rng_state"] > 2**53 for m in tokens_of(msgs)))


if __name__ == "__main__":
    unittest.main()
