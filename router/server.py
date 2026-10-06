import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_PORT = 8002
DEFAULT_WORKER_URL = os.environ.get("WORKER_URL", "http://localhost:8001")
WORKER_TIMEOUT = 5


class RouterHandler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, msg: str):
        self._send(code, json.dumps({"error": msg}).encode("utf-8"))

    def _forward(self, method: str):
        if not self.path.startswith("/snapshot"):
            self._send_error(404, "unknown path")
            return

        body = None
        if method == "POST":
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        req = Request(
            self.server.worker_url + self.path,
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(req, timeout=WORKER_TIMEOUT) as res:
                self._send(res.status, res.read())
        except HTTPError as e:
            # Worker answered with an error status; pass it through untouched
            with e:
                self._send(e.code, e.read())
        except TimeoutError:
            self._send_error(504, "worker timed out")
        except URLError as e:
            # urlopen wraps connect timeouts in URLError, read timeouts it raises bare
            if isinstance(e.reason, TimeoutError):
                self._send_error(504, "worker timed out")
            else:
                self._send_error(502, f"worker unreachable: {e.reason}")

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")


def make_server(port: int = DEFAULT_PORT, worker_url: str = DEFAULT_WORKER_URL, host: str = "127.0.0.1"):
    server = ThreadingHTTPServer((host, port), RouterHandler)
    server.worker_url = worker_url.rstrip("/")
    return server


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = make_server(port)
    print(f"router listening on :{port}, forwarding to {server.worker_url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
