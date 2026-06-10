from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_densenet.schedulers import (
    accumulation_window_example_count,
    accumulation_window_microbatch_count,
    optimizer_steps_per_epoch,
)


class DenseNetSchedulerTests(unittest.TestCase):
    def test_optimizer_steps_per_epoch_counts_accumulation_windows(self) -> None:
        self.assertEqual(optimizer_steps_per_epoch(28, 2), 14)
        self.assertEqual(optimizer_steps_per_epoch(29, 2), 15)

    def test_final_accumulation_window_uses_actual_microbatch_count(self) -> None:
        self.assertEqual(
            accumulation_window_microbatch_count(
                batch_index=27,
                total_batches=29,
                gradient_accumulation_steps=2,
            ),
            2,
        )
        self.assertEqual(
            accumulation_window_microbatch_count(
                batch_index=29,
                total_batches=29,
                gradient_accumulation_steps=2,
            ),
            1,
        )

    def test_final_accumulation_window_uses_actual_example_count(self) -> None:
        self.assertEqual(
            accumulation_window_example_count(
                batch_index=26,
                total_batches=28,
                total_examples=3565,
                batch_size=128,
                gradient_accumulation_steps=2,
            ),
            256,
        )
        self.assertEqual(
            accumulation_window_example_count(
                batch_index=27,
                total_batches=28,
                total_examples=3565,
                batch_size=128,
                gradient_accumulation_steps=2,
            ),
            237,
        )
        self.assertEqual(
            accumulation_window_example_count(
                batch_index=28,
                total_batches=28,
                total_examples=3565,
                batch_size=128,
                gradient_accumulation_steps=2,
            ),
            237,
        )


if __name__ == "__main__":
    unittest.main()
