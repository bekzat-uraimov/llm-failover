import unittest
import random
import os
from worker import snapshot


class SimpleModel:
    def __init__(self, seed=None):
        # seed may be None when restoring from state
        self.rng = random.Random(seed)
        self.tokens = []

    def gen_one(self):
        t = self.rng.randint(0, 99999)
        self.tokens.append(t)
        return t


class ResumeExperimentTest(unittest.TestCase):
    def setUp(self):
        # ensure snapshot dir exists
        os.makedirs(os.path.join(os.path.dirname(snapshot.__file__), "snapshots"), exist_ok=True)

    def test_resume_at_50_matches_original(self):
        m = SimpleModel(seed=42)
        tokens = []
        sid = None
        for i in range(100):
            tokens.append(m.gen_one())
            if i == 49:
                # save snapshot at position 50
                sid = snapshot.save_snapshot(tokens.copy(), m.rng.getstate(), i + 1, {"seed": 42})

        self.assertIsNotNone(sid)

        # load snapshot and resume in a fresh model
        th, rs, pos, meta = snapshot.load_snapshot(sid)
        self.assertEqual(pos, 50)
        self.assertEqual(len(th), 50)

        m2 = SimpleModel(seed=0)
        m2.tokens = th.copy()
        m2.rng.setstate(rs)

        continued = [m2.gen_one() for _ in range(50)]
        self.assertEqual(continued, tokens[50:100])


if __name__ == "__main__":
    unittest.main()
