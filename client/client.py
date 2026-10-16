"""Send a prompt to the router and print the story as it streams in.

    python3 -m client.client --prompt "Once upon a time" --seed 42 --steps 100
"""

import argparse
import json
import socket
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--prompt", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=256, help="forward passes, same as llama2.c's -n")
    args = p.parse_args()

    with socket.create_connection((args.host, args.port)) as sock:
        f = sock.makefile("rwb")
        req = {"type": "GENERATE", "prompt": args.prompt, "seed": args.seed, "steps": args.steps}
        f.write(json.dumps(req).encode() + b"\n")
        f.flush()

        for line in f:
            msg = json.loads(line)
            if msg["type"] == "TEXT":
                sys.stdout.write(msg["text"])
                sys.stdout.flush()
            elif msg["type"] == "END":
                break
            elif msg["type"] == "ERROR":
                sys.exit(f"\nerror: {msg['error']}")

    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
