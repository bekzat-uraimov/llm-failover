"""Send a prompt to a worker and print the story as tokens stream in.

    python3 -m client.client --prompt "Once upon a time" --seed 42 --steps 100
"""

import argparse
import json
import socket
import sys

from client.tokenizer import decode, load_vocab


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9001)
    p.add_argument("--prompt", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=256, help="forward passes, same as llama2.c's -n")
    p.add_argument("--tokenizer", default="models/tokenizer.bin")
    args = p.parse_args()

    vocab = load_vocab(args.tokenizer)
    out = sys.stdout.buffer

    with socket.create_connection((args.host, args.port)) as sock:
        f = sock.makefile("rwb")
        start = {"type": "START", "session": "cli", "epoch": 1,
                 "prompt": args.prompt, "seed": args.seed, "steps": args.steps}
        f.write(json.dumps(start).encode() + b"\n")
        f.flush()

        prev = None
        for line in f:
            msg = json.loads(line)
            if msg["type"] == "STARTED":
                # Print the prompt the way llama2.c does, decoded token by token
                tokens = msg["tokens"]
                for a, b in zip(tokens, tokens[1:]):
                    out.write(decode(vocab, a, b))
                prev = tokens[-1]
            elif msg["type"] == "TOKEN":
                out.write(decode(vocab, prev, msg["token"]))
                prev = msg["token"]
            elif msg["type"] == "DONE":
                break
            elif msg["type"] == "ERROR":
                sys.exit(f"worker error: {msg['error']}")
            out.flush()

    out.write(b"\n")


if __name__ == "__main__":
    main()
