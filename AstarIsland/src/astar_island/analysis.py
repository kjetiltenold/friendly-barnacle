from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .api import AstarIslandClient
from .baseline import MIN_PROBABILITY_FLOOR, ModelParameters, build_all_predictions
from .cache import CacheStore
from .historical import build_historical_signal_prior
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
    submitted_weighted_kl: float | None
    submitted_score: float | None
    official_score: float | None


@dataclass(slots=True)
class RoundEvaluation:
    round_id: str
    round_number: int
    seed_evaluations: list[SeedEvaluation]
    average_model_score: float
    average_submitted_score: float | None
    average_official_score: float | None


@dataclass(slots=True)
class PreflightResult:
    synced_rounds: list[dict[str, int | str]]
    case_count: int
    before_score: float | None
    after_score: float | None
    improved: bool
    saved_params: bool
    report_path: str | None = None


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
        submitted_weighted_kl, submitted_score = prediction_score(
            ground_truth,
            np.array(analysis.prediction, dtype=float),
        )
        observation_count = sum(1 for observation in case.observations if observation.seed_index == seed_index)
        seed_evaluations.append(
            SeedEvaluation(
                seed_index=seed_index,
                observation_count=observation_count,
                weighted_kl=weighted_kl,
                model_score=model_score,
                submitted_weighted_kl=submitted_weighted_kl,
                submitted_score=submitted_score,
                official_score=analysis.score,
            )
        )

    average_model_score = float(np.mean([item.model_score for item in seed_evaluations])) if seed_evaluations else 0.0
    submitted_scores = [item.submitted_score for item in seed_evaluations if item.submitted_score is not None]
    average_submitted_score = float(np.mean(submitted_scores)) if submitted_scores else None
    official_scores = [item.official_score for item in seed_evaluations if item.official_score is not None]
    average_official_score = float(np.mean(official_scores)) if official_scores else None
    return RoundEvaluation(
        round_id=case.round_id,
        round_number=case.round_number,
        seed_evaluations=seed_evaluations,
        average_model_score=average_model_score,
        average_submitted_score=average_submitted_score,
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
                "average_submitted_score": evaluation.average_submitted_score,
                "average_official_score": evaluation.average_official_score,
                "seeds": [asdict(seed_eval) for seed_eval in evaluation.seed_evaluations],
            }
            for evaluation in evaluations
        ],
    }


def load_cached_evaluation_cases(
    client: AstarIslandClient,
    cache: CacheStore,
    *,
    round_ids: list[str] | None = None,
    completed_only: bool = True,
) -> list[EvaluationCase]:
    candidate_ids = round_ids
    if candidate_ids is None:
        rounds = client.get_my_rounds()
        candidate_ids = [
            round_item.id
            for round_item in rounds
            if (round_item.status == "completed" if completed_only else True)
        ]

    cases: list[EvaluationCase] = []
    for round_id in candidate_ids:
        detail = cache.load_round_detail(round_id)
        analyses = cache.load_all_round_analyses(round_id)
        observations = cache.load_observations(round_id)
        if detail is None or not analyses or not observations:
            continue
        cases.append(
            EvaluationCase(
                round_id=round_id,
                round_number=detail.round_number,
                detail=detail,
                observations=observations,
                analyses=analyses,
            )
        )
    return cases


def sync_completed_analyses(
    client: AstarIslandClient,
    cache: CacheStore,
    *,
    round_ids: list[str] | None = None,
    include_non_completed: bool = False,
    refresh: bool = False,
) -> list[dict[str, int | str]]:
    requested_ids = round_ids

    if requested_ids is None:
        rounds = client.get_my_rounds()
        selected_rounds = [
            round_item
            for round_item in rounds
            if (round_item.status == "completed" if not include_non_completed else True)
        ]
        candidate_ids = [round_item.id for round_item in selected_rounds]
    else:
        candidate_ids = requested_ids

    synced: list[dict[str, int | str]] = []
    for round_id in candidate_ids:
        detail = cache.load_round_detail(round_id)
        if detail is None or refresh:
            detail = client.get_round_detail(round_id)
            cache.save_round_detail(detail)

        synced_seed_count = 0
        for seed_index in range(detail.seeds_count):
            if not refresh and cache.load_round_analysis(round_id, seed_index) is not None:
                continue
            analysis = client.get_round_analysis(round_id=round_id, seed_index=seed_index)
            cache.save_round_analysis(round_id, seed_index, analysis)
            synced_seed_count += 1

        if synced_seed_count > 0 or refresh:
            synced.append(
                {
                    "round_id": round_id,
                    "round_number": detail.round_number,
                    "seeds_count": detail.seeds_count,
                    "synced_seed_count": synced_seed_count,
                }
            )
    return synced


def run_training_preflight(
    client: AstarIslandClient,
    cache: CacheStore,
    *,
    passes: int = 1,
    round_ids: list[str] | None = None,
    include_non_completed: bool = False,
    refresh: bool = False,
) -> PreflightResult:
    synced_rounds = sync_completed_analyses(
        client,
        cache,
        round_ids=round_ids,
        include_non_completed=include_non_completed,
        refresh=refresh,
    )
    cases = load_cached_evaluation_cases(
        client,
        cache,
        round_ids=round_ids,
        completed_only=not include_non_completed,
    )
    if not cases:
        return PreflightResult(
            synced_rounds=synced_rounds,
            case_count=0,
            before_score=None,
            after_score=None,
            improved=False,
            saved_params=False,
        )

    current_params = cache.load_model_parameters() or ModelParameters()
    before_score = mean_model_score(cases, current_params)
    best_params, best_score, history = coordinate_search(cases, start=current_params, passes=passes)
    improved = best_score >= before_score
    saved_params = False
    if improved:
        cache.save_model_parameters(best_params)
        saved_params = True

    evaluations = evaluate_cases(cases, best_params if improved else current_params)
    historical_prior = build_historical_signal_prior(
        [(case.detail, case.analyses) for case in cases]
    )
    cache.save_historical_signal_prior(historical_prior)
    report = evaluation_report(evaluations, best_params if improved else current_params)
    report["preflight"] = {
        "before_score": before_score,
        "after_score": best_score,
        "improved": improved,
        "history": history,
        "synced_rounds": synced_rounds,
        "case_count": len(cases),
    }
    report_path = str(cache.save_report("preflight", report))
    return PreflightResult(
        synced_rounds=synced_rounds,
        case_count=len(cases),
        before_score=before_score,
        after_score=best_score,
        improved=improved,
        saved_params=saved_params,
        report_path=report_path,
    )
