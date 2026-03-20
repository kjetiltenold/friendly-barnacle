from __future__ import annotations

import unittest

import numpy as np

from astar_island.baseline import aggregate_observation_counts, build_prior, normalize_probabilities
from astar_island.types import InitialState, Settlement, SimulationResult, Viewport


class BaselineTests(unittest.TestCase):
    def test_normalize_applies_probability_floor(self) -> None:
        raw = np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]], dtype=float)
        normalized = normalize_probabilities(raw)
        self.assertAlmostEqual(float(normalized.sum()), 1.0)
        self.assertTrue(np.all(normalized >= 0.01))

    def test_build_prior_returns_expected_shape(self) -> None:
        state = InitialState(
            grid=[[10, 10, 10], [10, 11, 10], [10, 10, 10]],
            settlements=[Settlement(x=1, y=1, has_port=False, alive=True)],
        )
        prior = build_prior(state)
        self.assertEqual(prior.shape, (3, 3, 6))
        self.assertAlmostEqual(float(prior[1, 1].sum()), 1.0)

    def test_observation_counts_mark_coverage(self) -> None:
        observation = SimulationResult(
            round_id="round",
            seed_index=0,
            grid=[[1, 2], [3, 4]],
            settlements=[],
            viewport=Viewport(x=1, y=1, w=2, h=2),
            width=4,
            height=4,
            queries_used=1,
            queries_max=50,
        )
        counts, coverage = aggregate_observation_counts([observation], height=4, width=4)
        self.assertEqual(int(coverage.sum()), 4)
        self.assertEqual(int(counts[1, 1, 1]), 1)
        self.assertEqual(int(counts[1, 2, 2]), 1)


if __name__ == "__main__":
    unittest.main()
