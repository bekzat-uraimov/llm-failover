import unittest
import random
import os
from worker import snapshot


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        # ensure snapshot dir exists
        os.makedirs(os.path.join(os.path.dirname(snapshot.__file__), "snapshots"), exist_ok=True)

    def test_rng_roundtrip(self):
        r = random.Random(42)
        # advance RNG a bit
        seq1 = [r.randint(0, 1000000) for _ in range(10)]
        state = r.getstate()
        sid = snapshot.save_snapshot([], state, 10, {"note": "rng test"})

        th, loaded_state, pos, meta = snapshot.load_snapshot(sid)
        self.assertEqual(pos, 10)
        self.assertEqual(meta.get("note"), "rng test")

        r2 = random.Random()
        r2.setstate(loaded_state)
        seq2 = [r2.randint(0, 1000000) for _ in range(10)]
        # continue from state should produce new sequence; not equal to seq1
        self.assertNotEqual(seq1, seq2)

    def test_token_history_roundtrip(self):
        tokens = [1, 2, 3, 5, 8, 13]
        r = random.Random(7)
        sid = snapshot.save_snapshot(tokens, r.getstate(), len(tokens), {"model": "tiny"})
        th, rs, pos, meta = snapshot.load_snapshot(sid)
        self.assertEqual(th, tokens)
        self.assertEqual(pos, len(tokens))
        self.assertEqual(meta.get("model"), "tiny")

    def test_save_with_encoded_rng_state(self):
        r = random.Random(2)
        encoded = snapshot._encode_rng_state(r.getstate())
        sid = snapshot.save_snapshot([], encoded, 0, {"note": "encoded rng"})
        th, loaded_state, pos, meta = snapshot.load_snapshot(sid)
        self.assertEqual(pos, 0)
        # loaded_state should be a valid RNG state that we can set on a Random
        r2 = random.Random()
        r2.setstate(loaded_state)
        # sanity check: next value matches original RNG after advancing one
        r3 = random.Random()
        r3.setstate(r.getstate())
        self.assertEqual(r2.randint(0, 1000000), r3.randint(0, 1000000))


if __name__ == "__main__":
    unittest.main()
