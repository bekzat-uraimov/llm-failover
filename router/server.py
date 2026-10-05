import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import threading
import sys

WORKER_URL = "http://localhost:8001"


class RouterHandler(BaseHTTPRequestHandler):
    def _send_json(self, code, obj):
        bs = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(bs)))
        self.end_headers()
        self.wfile.write(bs)

    def _forward(self, path, body_bytes):
        url = WORKER_URL + path
        req = Request(url, data=body_bytes, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=5) as resp:
                return resp.getcode(), resp.read()
        except HTTPError as e:
            return e.code, e.read() if hasattr(e, 'read') else b''
        except URLError as e:
            return 502, json.dumps({"error": str(e)}).encode('utf-8')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        p = self.path
        if p in ("/snapshot/save", "/snapshot/load"):
            code, resp_body = self._forward(p, body)
            try:
                obj = json.loads(resp_body.decode('utf-8')) if resp_body else {}
            except Exception:
                obj = {"raw": resp_body.decode('utf-8', errors='replace')}
            self._send_json(code, obj)
            return
        self._send_json(404, {"error": "unknown"})


def serve(port: int = 8002):
    server = HTTPServer(("", port), RouterHandler)
    print(f"Router HTTP server listening on :{port} -> forwards to {WORKER_URL}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    p = 8002
    if len(sys.argv) > 1:
        try:
            p = int(sys.argv[1])
        except Exception:
            pass
    serve(p)
