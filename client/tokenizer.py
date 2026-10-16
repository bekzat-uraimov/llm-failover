"""Token id -> text, matching llama2.c's decode() and safe_printf()."""

import re
import struct

BOS = 1

_BYTE_TOKEN = re.compile(rb"<0x([0-9A-Fa-f]{2})>")


def load_vocab(path: str) -> list[bytes]:
    # tokenizer.bin: int32 max_token_length, then per token: float32 score, int32 length, bytes
    with open(path, "rb") as f:
        data = f.read()
    vocab = []
    off = 4
    while off < len(data):
        _, n = struct.unpack_from("<fi", data, off)
        off += 8
        vocab.append(data[off:off + n])
        off += n
    return vocab


def decode(vocab: list[bytes], prev: int, token: int) -> bytes:
    piece = vocab[token]
    # After BOS, sentencepiece strips the leading space
    if prev == BOS and piece.startswith(b" "):
        piece = piece[1:]
    m = _BYTE_TOKEN.match(piece)
    if m:
        piece = bytes([int(m.group(1), 16)])
    # Lone control bytes (backspace etc.) are dropped, like llama2.c's safe_printf
    if len(piece) == 1 and not (0x20 <= piece[0] < 0x7F or piece in b" \t\n\v\f\r"):
        return b""
    return piece
