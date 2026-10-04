import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from urllib.parse import urlparse
import os
from . import snapshot
import sys


class SimpleHandler(BaseHTTPRequestHandler):
    def _send_json(self, code, obj):
        bs = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(bs)))
        self.end_headers()
        self.wfile.write(bs)

    def do_GET(self):
        p = urlparse(self.path)
        if p.path.startswith("/snapshot/"):
            sid = p.path[len("/snapshot/"):]
            try:
                token_history, rng_state, position, metadata = snapshot.load_snapshot(sid)
            except FileNotFoundError:
                self._send_json(404, {"error": "not found"})
                return
            # Do not return raw rng_state bytes in JSON; return metadata and length
            self._send_json(200, {"id": sid, "position": position, "tokens": len(token_history), "metadata": metadata})
            return
        if p.path == "/snapshot":
            ids = snapshot.list_snapshots()
            self._send_json(200, {"snapshots": ids})
            return
        self._send_json(404, {"error": "unknown"})

    def do_POST(self):
        p = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        if p.path == "/snapshot/save":
            try:
                req = json.loads(body.decode("utf-8")) if body else {}
                token_history = req.get("token_history", [])
                rng_state = req.get("rng_state")
                # If rng_state provided as base64 pickled bytes, accept; otherwise expect None
                if rng_state is None:
                    rng_state = None
                position = req.get("position", len(token_history))
                metadata = req.get("metadata", {})
                sid = snapshot.save_snapshot(token_history, rng_state, position, metadata)
                self._send_json(200, {"id": sid})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return
        if p.path == "/snapshot/load":
            try:
                req = json.loads(body.decode("utf-8")) if body else {}
                sid = req.get("id")
                if not sid:
                    self._send_json(400, {"error": "missing id"})
                    return
                token_history, rng_state, position, metadata = snapshot.load_snapshot(sid)
                # We return token history and position; rng_state is not returned raw
                self._send_json(200, {"id": sid, "position": position, "tokens": token_history, "metadata": metadata})
            except FileNotFoundError:
                self._send_json(404, {"error": "not found"})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return
        self._send_json(404, {"error": "unknown"})


def serve(port: int = 8000):
    server = HTTPServer(("", port), SimpleHandler)
    print(f"Worker HTTP server listening on :{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    p = 8000
    if len(sys.argv) > 1:
        try:
            p = int(sys.argv[1])
        except Exception:
            pass
    serve(p)
