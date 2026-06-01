from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments/hand_segmentor"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(EXPERIMENT_DIR))
common = load_module("common", "common.py")
supervised = load_module("hand_segmentor_supervised", "train_supervised_segmentor.py")


class HandSegmentorExperimentTests(unittest.TestCase):
    def test_video_level_splits_are_disjoint(self):
        names = list(common.SPLITS)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                overlap = set(common.SPLITS[left_name]) & set(common.SPLITS[right_name])
                self.assertFalse(overlap, f"{left_name}/{right_name} overlap: {overlap}")

    def test_existing_local_train_video_has_hand_frames(self):
        frames = common.split_frame_keys("train", hand_only=True)
        p06_frames = [frame for frame in frames if frame.video_id == "P06_110"]
        self.assertGreater(len(p06_frames), 0)
        for frame in p06_frames:
            self.assertTrue(common.sparse_frame_path(frame).exists())

    def test_test_split_uses_whole_unseen_videos(self):
        self.assertIn("P37_102", common.SPLITS["val"])
        self.assertNotIn("P37_102", common.SPLITS["train"])
        self.assertNotIn("P37_102", common.SPLITS["test"])
        self.assertIn("P22_107", common.SPLITS["test"])
        self.assertNotIn("P22_107", common.SPLITS["train"])
        self.assertNotIn("P22_107", common.SPLITS["val"])

    def test_supervised_helpers(self):
        self.assertEqual(supervised.parse_size("512x912"), (512, 912))
        keys = [common.FrameKey("P01_01", 1), common.FrameKey("P01_01", 2), common.FrameKey("P02_01", 1)]
        self.assertEqual(supervised.count_by_video(keys), {"P01_01": 2, "P02_01": 1})


if __name__ == "__main__":
    unittest.main()
