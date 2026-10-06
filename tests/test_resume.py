import random
import tempfile
import unittest

from worker import snapshot


class FakeModel:
    """Stand-in for the sampler: each token is one RNG draw, so only RNG state matters."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.tokens = []

    def next_token(self) -> int:
        t = self.rng.randint(0, 31999)
        self.tokens.append(t)
        return t


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_dir = snapshot.SNAPSHOT_DIR
        snapshot.SNAPSHOT_DIR = self.tmp.name

    def tearDown(self):
        snapshot.SNAPSHOT_DIR = self.old_dir
        self.tmp.cleanup()

    def test_resume_at_50_matches_uninterrupted_run(self):
        m = FakeModel(seed=42)
        for _ in range(50):
            m.next_token()
        sid = snapshot.save_snapshot(m.tokens, m.rng.getstate(), 50, {"seed": 42})
        for _ in range(50):
            m.next_token()

        tokens, state, pos, _ = snapshot.load_snapshot(sid)
        m2 = FakeModel()
        m2.tokens = tokens
        m2.rng.setstate(state)
        for _ in range(100 - pos):
            m2.next_token()

        self.assertEqual(m2.tokens, m.tokens)


if __name__ == "__main__":
    unittest.main()
