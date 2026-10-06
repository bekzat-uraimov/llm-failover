import os
import random
import tempfile
import unittest

from worker import snapshot


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_dir = snapshot.SNAPSHOT_DIR
        snapshot.SNAPSHOT_DIR = self.tmp.name

    def tearDown(self):
        snapshot.SNAPSHOT_DIR = self.old_dir
        self.tmp.cleanup()

    def test_rng_state_roundtrip_continues_same_sequence(self):
        r = random.Random(42)
        [r.random() for _ in range(10)]
        sid = snapshot.save_snapshot([], r.getstate(), 0)

        _, state, _, _ = snapshot.load_snapshot(sid)
        r2 = random.Random()
        r2.setstate(state)

        self.assertEqual([r2.random() for _ in range(10)], [r.random() for _ in range(10)])

    def test_token_history_and_metadata_roundtrip(self):
        tokens = [1, 2, 3, 5, 8, 13]
        sid = snapshot.save_snapshot(tokens, random.Random(7).getstate(), len(tokens), {"model": "tiny"})

        th, _, pos, meta = snapshot.load_snapshot(sid)

        self.assertEqual(th, tokens)
        self.assertEqual(pos, 6)
        self.assertEqual(meta, {"model": "tiny"})

    def test_list_snapshots(self):
        sid = snapshot.save_snapshot([1], random.Random(1).getstate(), 1)
        self.assertEqual(snapshot.list_snapshots(), [sid])

    def test_save_leaves_no_temp_files(self):
        snapshot.save_snapshot([1], random.Random(1).getstate(), 1)
        self.assertFalse([f for f in os.listdir(self.tmp.name) if f.endswith(".tmp")])

    def test_rejects_invalid_rng_state(self):
        for bad in [None, "abc", [3, [1, 2, 3], None]]:
            with self.assertRaises(ValueError):
                snapshot.save_snapshot([], bad, 0)

    def test_rejects_position_past_history(self):
        with self.assertRaises(ValueError):
            snapshot.save_snapshot([1, 2], random.Random(1).getstate(), 3)

    def test_rejects_path_traversal_id(self):
        with self.assertRaises(ValueError):
            snapshot.load_snapshot("../../etc/passwd")

    def test_missing_snapshot_raises_not_found(self):
        with self.assertRaises(FileNotFoundError):
            snapshot.load_snapshot("0" * 32)


if __name__ == "__main__":
    unittest.main()
