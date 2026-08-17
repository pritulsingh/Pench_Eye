"""Unit and sanity tests for Blank Image Triage classification logic."""
import unittest

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python-headless is required for ML triage tests")

from ml.triage.blank_classifier import BlankImageClassifier  # noqa: E402
from ml.triage.features import ImageFeatureExtractor  # noqa: E402


class TestTriageLogic(unittest.TestCase):
    def setUp(self):
        self.classifier = BlankImageClassifier(ml_mode="demo", blank_threshold=0.95)
        self.extractor = ImageFeatureExtractor()

    def test_solid_black_is_blank(self):
        black_img = np.zeros((100, 100, 3), dtype=np.uint8)
        res = self.classifier.classify(black_img)
        self.assertTrue(res.is_blank)
        self.assertGreaterEqual(res.blank_probability, 0.95)

    def test_solid_white_is_blank(self):
        white_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        res = self.classifier.classify(white_img)
        self.assertTrue(res.is_blank)
        self.assertGreaterEqual(res.blank_probability, 0.95)

    def test_pattern_image_is_not_blank(self):
        # Create patterned image with sharp edges
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[::20, :] = 255
        img[:, ::20] = 200
        res = self.classifier.classify(img)
        self.assertFalse(res.is_blank)
        self.assertLess(res.blank_probability, 0.95)

    def test_feature_extractor_bounds(self):
        img = np.random.randint(0, 256, (120, 120, 3), dtype=np.uint8)
        features = self.extractor.extract(img)
        self.assertGreaterEqual(features.brightness, 0.0)
        self.assertLessEqual(features.brightness, 1.0)
        self.assertGreaterEqual(features.quality_score, 0.0)
        self.assertLessEqual(features.quality_score, 1.0)


if __name__ == '__main__':
    unittest.main()
