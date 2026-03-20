from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .baseline import MIN_PROBABILITY_FLOOR, ModelParameters, build_all_predictions
from .types import RoundAnalysis, RoundDetail, SimulationResult


@dataclass(slots=True)
class EvaluationCase:
    round_id: str
    round_number: int
    detail: RoundDetail
    observations: list[SimulationResult]
    analyses: dict[int, RoundAnalysis]


@dataclass(slots=True)
class SeedEvaluation:
    seed_index: int
    observation_count: int
    weighted_kl: float
    model_score: float
    official_score: float | None


@dataclass(slots=True)
class RoundEvaluation:
    round_id: str
    round_number: int
    seed_evaluations: list[SeedEvaluation]
    average_model_score: float
    average_official_score: float | None


def prediction_score(ground_truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    p = np.maximum(ground_truth.astype(float), 0.0)
    q = np.maximum(prediction.astype(float), MIN_PROBABILITY_FLOOR * 0.1)

    p_positive = p > 0.0
    safe_p = np.where(p_positive, p, 1.0)
    entropy = -np.sum(np.where(p_positive, p * np.log(safe_p), 0.0), axis=-1)
    kl = np.sum(np.where(p_positive, p * (np.log(safe_p) - np.log(q)), 0.0), axis=-1)

    total_entropy = float(np.sum(entropy))
    if total_entropy <= 0.0:
        return 0.0, 100.0

    weighted_kl = float(np.sum(entropy * kl) / total_entropy)
    score = max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))
    return weighted_kl, float(score)


def evaluate_case(case: EvaluationCase, params: ModelParameters) -> RoundEvaluation:
    predictions = build_all_predictions(case.detail, case.observations, params=params)
    seed_evaluations: list[SeedEvaluation] = []

    for seed_index in sorted(case.analyses):
        analysis = case.analyses[seed_index]
        predicted = np.array(predictions[seed_index].prediction, dtype=float)
        ground_truth = np.array(analysis.ground_truth, dtype=float)
        weighted_kl, model_score = prediction_score(ground_truth, predicted)
        observation_count = sum(1 for observation in case.observations if observation.seed_index == seed_index)
        seed_evaluations.append(
            SeedEvaluation(
                seed_index=seed_index,
                observation_count=observation_count,
                weighted_kl=weighted_kl,
                model_score=model_score,
                official_score=analysis.score,
            )
        )

    average_model_score = float(np.mean([item.model_score for item in seed_evaluations])) if seed_evaluations else 0.0
    official_scores = [item.official_score for item in seed_evaluations if item.official_score is not None]
    average_official_score = float(np.mean(official_scores)) if official_scores else None
    return RoundEvaluation(
        round_id=case.round_id,
        round_number=case.round_number,
        seed_evaluations=seed_evaluations,
        average_model_score=average_model_score,
        average_official_score=average_official_score,
    )


def evaluate_cases(cases: list[EvaluationCase], params: ModelParameters) -> list[RoundEvaluation]:
    return [evaluate_case(case, params) for case in cases]


def mean_model_score(cases: list[EvaluationCase], params: ModelParameters) -> float:
    evaluations = evaluate_cases(cases, params)
    seed_scores = [
        seed_eval.model_score
        for round_eval in evaluations
        for seed_eval in round_eval.seed_evaluations
    ]
    if not seed_scores:
        return 0.0
    return float(np.mean(seed_scores))


def default_search_space() -> dict[str, list[float]]:
    return {
        "prior_weight": [1.5, 2.0, 2.5, 3.0, 3.5],
        "exact_weight": [4.5, 6.0, 7.5, 9.0],
        "spatial_weight": [0.5, 0.75, 1.0, 1.25, 1.5],
        "global_scale": [0.5, 1.0, 1.5],
        "code_scale": [0.5, 0.9, 1.3],
        "frontier_scale": [0.7, 1.1, 1.5],
        "context_scale": [0.8, 1.2, 1.6],
        "region_scale": [0.5, 0.9, 1.3],
    }


def coordinate_search(
    cases: list[EvaluationCase],
    *,
    start: ModelParameters | None = None,
    passes: int = 2,
    search_space: dict[str, list[float]] | None = None,
) -> tuple[ModelParameters, float, list[dict[str, float | int | str]]]:
    params = start or ModelParameters()
    space = search_space or default_search_space()
    best_score = mean_model_score(cases, params)
    history: list[dict[str, float | int | str]] = []

    for pass_index in range(passes):
        improved = False
        for name, values in space.items():
            local_best_params = params
            local_best_score = best_score
            for value in values:
                candidate = replace(params, **{name: value})
                score = mean_model_score(cases, candidate)
                history.append(
                    {
                        "pass": pass_index,
                        "parameter": name,
                        "value": value,
                        "score": score,
                    }
                )
                if score > local_best_score:
                    local_best_score = score
                    local_best_params = candidate
            if local_best_score > best_score:
                params = local_best_params
                best_score = local_best_score
                improved = True
        if not improved:
            break

    return params, best_score, history


def evaluation_report(evaluations: list[RoundEvaluation], params: ModelParameters) -> dict[str, object]:
    return {
        "model_parameters": asdict(params),
        "rounds": [
            {
                "round_id": evaluation.round_id,
                "round_number": evaluation.round_number,
                "average_model_score": evaluation.average_model_score,
                "average_official_score": evaluation.average_official_score,
                "seeds": [asdict(seed_eval) for seed_eval in evaluation.seed_evaluations],
            }
            for evaluation in evaluations
        ],
    }
