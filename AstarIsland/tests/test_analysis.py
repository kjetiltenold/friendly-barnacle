from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from astar_island.analysis import EvaluationCase, coordinate_search, evaluate_case, prediction_score
from astar_island.baseline import ModelParameters, build_all_predictions
from astar_island.cache import CacheStore
from astar_island.types import InitialState, RoundAnalysis, RoundDetail, Settlement, SimulationResult, Viewport


def make_case() -> EvaluationCase:
    detail = RoundDetail(
        id="round",
        round_number=7,
        status="completed",
        map_width=4,
        map_height=4,
        seeds_count=1,
        initial_states=[
            InitialState(
                grid=[
                    [10, 10, 10, 10],
                    [10, 11, 11, 10],
                    [10, 11, 4, 10],
                    [10, 10, 10, 10],
                ],
                settlements=[Settlement(x=1, y=1, has_port=False, alive=True)],
            )
        ],
    )
    observations = [
        SimulationResult(
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
    ]
    predictions = build_all_predictions(detail, observations)
    analysis = RoundAnalysis(
        prediction=predictions[0].prediction.tolist(),
        ground_truth=predictions[0].prediction.tolist(),
        score=100.0,
        width=4,
        height=4,
        initial_grid=detail.initial_states[0].grid,
    )
    return EvaluationCase(
        round_id=detail.id,
        round_number=detail.round_number,
        detail=detail,
        observations=observations,
        analyses={0: analysis},
    )


class AnalysisTests(unittest.TestCase):
    def test_prediction_score_is_perfect_for_identical_tensors(self) -> None:
        tensor = np.array([[[0.7, 0.1, 0.1, 0.05, 0.03, 0.02]]], dtype=float)
        weighted_kl, score = prediction_score(tensor, tensor)
        self.assertAlmostEqual(weighted_kl, 0.0, places=9)
        self.assertAlmostEqual(score, 100.0, places=9)

    def test_evaluate_case_matches_perfect_ground_truth(self) -> None:
        case = make_case()
        evaluation = evaluate_case(case, ModelParameters())
        self.assertAlmostEqual(evaluation.average_model_score, 100.0, places=6)
        self.assertAlmostEqual(evaluation.average_official_score or 0.0, 100.0, places=6)

    def test_coordinate_search_runs_and_returns_parameters(self) -> None:
        case = make_case()
        params, best_score, history = coordinate_search([case], passes=1)
        self.assertIsInstance(params, ModelParameters)
        self.assertGreaterEqual(best_score, 99.0)
        self.assertTrue(history)

    def test_cache_round_analysis_and_model_parameters(self) -> None:
        case = make_case()
        analysis = case.analyses[0]
        params = ModelParameters(prior_weight=3.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheStore(Path(tmpdir))
            cache.save_round_analysis(case.round_id, 0, analysis)
            cache.save_model_parameters(params)

            loaded_analysis = cache.load_round_analysis(case.round_id, 0)
            loaded_params = cache.load_model_parameters()

            self.assertIsNotNone(loaded_analysis)
            self.assertEqual(loaded_analysis.width if loaded_analysis else None, 4)
            self.assertIsNotNone(loaded_params)
            self.assertAlmostEqual((loaded_params or ModelParameters()).prior_weight, 3.0)


if __name__ == "__main__":
    unittest.main()
