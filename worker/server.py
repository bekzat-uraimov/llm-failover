import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import snapshot

DEFAULT_PORT = 8001


class WorkerHandler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(req, dict):
            raise ValueError("request body must be a JSON object")
        return req

    def do_GET(self):
        if self.path == "/snapshot":
            self._send_json(200, {"snapshots": snapshot.list_snapshots()})
            return
        if self.path.startswith("/snapshot/"):
            sid = self.path[len("/snapshot/"):]
            try:
                tokens, _, position, metadata = snapshot.load_snapshot(sid)
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            except FileNotFoundError:
                self._send_json(404, {"error": "not found"})
                return
            self._send_json(200, {"id": sid, "position": position, "tokens": len(tokens), "metadata": metadata})
            return
        self._send_json(404, {"error": "unknown path"})

    def do_POST(self):
        if self.path == "/snapshot/save":
            try:
                req = self._read_json()
                if "rng_state" not in req:
                    raise ValueError("missing rng_state")
                tokens = req.get("token_history", [])
                sid = snapshot.save_snapshot(
                    tokens, req["rng_state"], req.get("position", len(tokens)), req.get("metadata")
                )
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            self._send_json(200, {"id": sid})
            return
        if self.path == "/snapshot/load":
            try:
                sid = self._read_json().get("id")
                tokens, rng_state, position, metadata = snapshot.load_snapshot(sid)
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            except FileNotFoundError:
                self._send_json(404, {"error": "not found"})
                return
            self._send_json(200, {
                "id": sid,
                "token_history": tokens,
                "rng_state": rng_state,
                "position": position,
                "metadata": metadata,
            })
            return
        self._send_json(404, {"error": "unknown path"})


def make_server(port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), WorkerHandler)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = make_server(port)
    print(f"worker listening on :{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
