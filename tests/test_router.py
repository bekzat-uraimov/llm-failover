import unittest
import threading
import time
import json
from worker import server as worker_server
from router import server as router_server
import urllib.request


def start_worker():
    worker_server.serve(8001)


def start_router():
    router_server.serve(8002)


class RouterIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wt = threading.Thread(target=start_worker, daemon=True)
        cls.wt.start()
        cls.rt = threading.Thread(target=start_router, daemon=True)
        cls.rt.start()
        time.sleep(0.2)

    def test_save_and_load_via_router(self):
        url = "http://localhost:8002/snapshot/save"
        payload = json.dumps({"token_history": [10, 20, 30], "position": 3}).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode('utf-8'))
        self.assertIn('id', body)
        sid = body['id']

        # load via router
        url2 = "http://localhost:8002/snapshot/load"
        payload2 = json.dumps({"id": sid}).encode('utf-8')
        req2 = urllib.request.Request(url2, data=payload2, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req2) as resp2:
            body2 = json.loads(resp2.read().decode('utf-8'))
        self.assertEqual(body2.get('tokens'), [10, 20, 30])


if __name__ == "__main__":
    unittest.main()
