import json
import random
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from router import server as router_server
from worker import server as worker_server
from worker import snapshot


def call(url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=5) as res:
            return res.status, json.loads(res.read())
    except HTTPError as e:
        with e:
            return e.code, json.loads(e.read())


def start(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def stop(server):
    server.shutdown()
    server.server_close()


class RouterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.old_dir = snapshot.SNAPSHOT_DIR
        snapshot.SNAPSHOT_DIR = cls.tmp.name

        # Port 0 lets the OS pick free ports, so tests don't collide with anything running locally
        cls.worker = start(worker_server.make_server(0))
        worker_url = f"http://127.0.0.1:{cls.worker.server_port}"
        cls.router = start(router_server.make_server(0, worker_url))
        cls.url = f"http://127.0.0.1:{cls.router.server_port}"

    @classmethod
    def tearDownClass(cls):
        stop(cls.router)
        stop(cls.worker)
        snapshot.SNAPSHOT_DIR = cls.old_dir
        cls.tmp.cleanup()

    def save(self, tokens: list[int], rng: random.Random) -> str:
        code, body = call(self.url + "/snapshot/save", {"token_history": tokens, "rng_state": rng.getstate()})
        self.assertEqual(code, 200, body)
        return body["id"]

    def test_save_and_load_through_router_resumes_rng(self):
        rng = random.Random(42)
        sid = self.save([10, 20, 30], rng)

        code, body = call(self.url + "/snapshot/load", {"id": sid})

        self.assertEqual(code, 200)
        self.assertEqual(body["token_history"], [10, 20, 30])
        self.assertEqual(body["position"], 3)
        resumed = random.Random()
        resumed.setstate(snapshot.parse_rng_state(body["rng_state"]))
        self.assertEqual(resumed.random(), rng.random())

    def test_get_snapshot_metadata(self):
        sid = self.save([1, 2], random.Random(1))

        code, body = call(self.url + f"/snapshot/{sid}")

        self.assertEqual(code, 200)
        self.assertEqual(body["tokens"], 2)

    def test_save_without_rng_state_is_rejected(self):
        code, body = call(self.url + "/snapshot/save", {"token_history": [1]})
        self.assertEqual(code, 400)
        self.assertIn("rng_state", body["error"])

    def test_load_with_bad_id_is_rejected(self):
        code, _ = call(self.url + "/snapshot/load", {"id": "../../secrets"})
        self.assertEqual(code, 400)

    def test_load_unknown_id_is_not_found(self):
        code, _ = call(self.url + "/snapshot/load", {"id": "0" * 32})
        self.assertEqual(code, 404)

    def test_unknown_path_is_not_found(self):
        code, _ = call(self.url + "/generate", {})
        self.assertEqual(code, 404)


class RouterWorkerDownTest(unittest.TestCase):
    def test_returns_502_when_worker_unreachable(self):
        # Grab a free port, then close it so nothing is listening there
        dead = worker_server.make_server(0)
        dead_url = f"http://127.0.0.1:{dead.server_port}"
        dead.server_close()
        router = start(router_server.make_server(0, dead_url))
        try:
            code, body = call(f"http://127.0.0.1:{router.server_port}/snapshot")
        finally:
            stop(router)

        self.assertEqual(code, 502)
        self.assertIn("unreachable", body["error"])


if __name__ == "__main__":
    unittest.main()
