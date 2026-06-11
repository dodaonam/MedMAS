from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_densenet.artifacts import DISEASE_LABELS, RARE_SAMPLER_WEAKCROP_VARIANT, SUPPORTED_RECIPE_VARIANTS
from train_densenet.sampling import rare_label_sampler_config_and_weights
from train_densenet.transforms import (
    LESION_PRESERVING_TRAIN_TRANSFORM,
    build_train_transform,
    train_transform_variant_for_recipe,
)


class DenseNetVariantTests(unittest.TestCase):
    def test_rare_sampler_uses_capped_max_row_weight(self) -> None:
        frame = pd.DataFrame(
            {
                "Infiltration": [1, 1, 1, 1, 0, 0],
                "Effusion": [0, 0, 0, 0, 0, 0],
                "Atelectasis": [0, 0, 0, 0, 0, 0],
                "Nodule": [0, 0, 0, 0, 1, 1],
                "Mass": [0, 0, 0, 0, 0, 1],
            }
        )
        config, weights = rare_label_sampler_config_and_weights(frame, labels=DISEASE_LABELS)

        self.assertTrue(config["rare_sampler_enabled"])
        self.assertEqual(config["reference_label"], "Infiltration")
        self.assertEqual(config["reference_positive_count"], 4)
        self.assertAlmostEqual(config["rare_sampler_boost"]["Nodule"], np.sqrt(4 / 2))
        self.assertEqual(config["rare_sampler_boost"]["Mass"], 2.0)
        self.assertEqual(weights[:4].tolist(), [1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(float(weights[4]), np.sqrt(4 / 2))
        self.assertEqual(float(weights[5]), 2.0)
        self.assertEqual(config["sample_weight_stats"]["weighted_row_count"], 2)
        self.assertIn("expected_sampled_positive_counts_per_epoch", config)

    def test_recipe_variant_is_named_and_explicit(self) -> None:
        self.assertIn(RARE_SAMPLER_WEAKCROP_VARIANT, SUPPORTED_RECIPE_VARIANTS)
        with self.assertRaisesRegex(ValueError, "Unsupported recipe_variant"):
            train_transform_variant_for_recipe("ad_hoc")

    def test_recipe_variant_selects_lesion_preserving_transform(self) -> None:
        self.assertEqual(
            train_transform_variant_for_recipe(RARE_SAMPLER_WEAKCROP_VARIANT),
            LESION_PRESERVING_TRAIN_TRANSFORM,
        )

    def test_lesion_preserving_transform_shape(self) -> None:
        try:
            transform = build_train_transform(input_size=320, resize_size=352, variant=LESION_PRESERVING_TRAIN_TRANSFORM)
        except ModuleNotFoundError:
            self.skipTest("torchvision is not installed")
        names = [step.__class__.__name__ for step in transform.transforms]
        self.assertEqual(names, ["Resize", "CenterCrop", "RandomRotation", "ColorJitter", "ToTensor", "Normalize"])


if __name__ == "__main__":
    unittest.main()
