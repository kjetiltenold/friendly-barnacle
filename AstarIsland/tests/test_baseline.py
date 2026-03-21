from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from astar_island.baseline import (
    aggregate_observation_counts,
    build_all_predictions,
    build_prior,
    infer_round_latent_profile,
    normalize_probabilities,
)
from astar_island.cache import CacheStore
from astar_island.historical import build_historical_signal_prior
from astar_island.planner import determine_stage, plan_query_batch, run_iterative_autopilot
from astar_island.types import InitialState, RoundAnalysis, RoundDetail, Settlement, SimulationResult, Viewport


def make_round_detail() -> RoundDetail:
    initial_states = [
        InitialState(
            grid=[
                [10, 10, 10, 10],
                [10, 11, 11, 10],
                [10, 11, 4, 10],
                [10, 10, 10, 10],
            ],
            settlements=[Settlement(x=1, y=1, has_port=False, alive=True)],
        ),
        InitialState(
            grid=[
                [10, 10, 10, 10],
                [10, 11, 11, 10],
                [10, 11, 11, 10],
                [10, 10, 10, 10],
            ],
            settlements=[Settlement(x=2, y=2, has_port=True, alive=True)],
        ),
    ]
    return RoundDetail(
        id="round",
        round_number=1,
        status="active",
        map_width=4,
        map_height=4,
        seeds_count=2,
        initial_states=initial_states,
    )


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

    def test_observation_counts_mark_coverage_and_samples(self) -> None:
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
        counts, coverage, samples = aggregate_observation_counts([observation], height=4, width=4)
        self.assertEqual(int(coverage.sum()), 4)
        self.assertEqual(int(samples.sum()), 4)
        self.assertEqual(int(counts[1, 1, 1]), 1)
        self.assertEqual(int(counts[1, 2, 2]), 1)

    def test_prediction_preview_includes_entropy_support_and_samples(self) -> None:
        detail = make_round_detail()
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
        predictions = build_all_predictions(detail, [observation])
        preview = predictions[0]
        self.assertEqual(preview.prediction.shape, (4, 4, 6))
        self.assertEqual(preview.entropy_grid.shape, (4, 4))
        self.assertEqual(preview.support_grid.shape, (4, 4))
        self.assertEqual(preview.sample_count_grid.shape, (4, 4))
        self.assertGreater(float(preview.sample_count_grid[1, 1]), 0.0)

    def test_latent_profile_reflects_observed_ruin_pressure(self) -> None:
        detail = make_round_detail()
        observation = SimulationResult(
            round_id="round",
            seed_index=0,
            grid=[[3, 3], [3, 3]],
            settlements=[],
            viewport=Viewport(x=1, y=1, w=2, h=2),
            width=4,
            height=4,
            queries_used=1,
            queries_max=50,
        )
        profile = infer_round_latent_profile(detail, [observation])
        self.assertGreater(profile.collapse_bias, 0.0)
        self.assertGreater(profile.certainty, 0.0)

    def test_planner_returns_ranked_queries(self) -> None:
        detail = make_round_detail()
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
        plan = plan_query_batch(detail, [observation], count=2, viewport_w=2, viewport_h=2)
        self.assertLessEqual(len(plan), 2)
        self.assertTrue(all(item.adjusted_score >= 0.0 for item in plan))
        self.assertTrue(all(0 <= item.x <= 2 for item in plan))
        self.assertTrue(all(0 <= item.y <= 2 for item in plan))

    def test_planner_can_use_historical_entropy_prior(self) -> None:
        detail = make_round_detail()
        ground_truth = np.full((4, 4, 6), [0.95, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=float)
        ground_truth[2, 2] = np.array([0.20, 0.20, 0.20, 0.20, 0.10, 0.10], dtype=float)
        prior = build_historical_signal_prior(
            [
                (
                    detail,
                    {
                        0: RoundAnalysis(
                            prediction=ground_truth.tolist(),
                            ground_truth=ground_truth.tolist(),
                            score=100.0,
                            width=4,
                            height=4,
                            initial_grid=detail.initial_states[0].grid,
                        )
                    },
                )
            ]
        )

        plan = plan_query_batch(
            detail,
            [],
            count=1,
            viewport_w=1,
            viewport_h=1,
            seed_indices=[0],
            historical_prior=prior,
        )
        self.assertEqual(len(plan), 1)
        self.assertEqual((plan[0].x, plan[0].y), (2, 2))

    def test_staged_planner_balances_early_exploration_across_seeds(self) -> None:
        detail = make_round_detail()
        plan = plan_query_batch(detail, [], count=2, viewport_w=2, viewport_h=2, stage="explore")
        self.assertEqual(len(plan), 2)
        self.assertEqual({item.seed_index for item in plan}, {0, 1})
        self.assertTrue(all(item.stage == "explore" for item in plan))

    def test_determine_stage_uses_progression(self) -> None:
        self.assertEqual(determine_stage(existing_queries=0, requested_queries=50), "explore")
        self.assertEqual(determine_stage(existing_queries=15, requested_queries=35), "infer")
        self.assertEqual(determine_stage(existing_queries=40, requested_queries=10), "exploit")

    def test_iterative_autopilot_executes_and_caches_results(self) -> None:
        detail = make_round_detail()

        class FakeClient:
            def simulate(
                self,
                *,
                round_id: str,
                seed_index: int,
                viewport_x: int,
                viewport_y: int,
                viewport_w: int,
                viewport_h: int,
            ) -> SimulationResult:
                grid = [[1 for _ in range(viewport_w)] for _ in range(viewport_h)]
                return SimulationResult(
                    round_id=round_id,
                    seed_index=seed_index,
                    grid=grid,
                    settlements=[],
                    viewport=Viewport(x=viewport_x, y=viewport_y, w=viewport_w, h=viewport_h),
                    width=detail.map_width,
                    height=detail.map_height,
                    queries_used=1,
                    queries_max=50,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheStore(Path(tmpdir))
            run = run_iterative_autopilot(
                FakeClient(),
                cache,
                detail,
                round_id=detail.id,
                total_queries=3,
                viewport_w=2,
                viewport_h=2,
                replan_every=1,
            )
            self.assertEqual(run.executed_queries, 3)
            self.assertEqual(len(cache.load_observations(detail.id)), 3)
            self.assertTrue(run.batch_stages)
            self.assertEqual(run.batch_stages[0], "explore")


if __name__ == "__main__":
    unittest.main()
