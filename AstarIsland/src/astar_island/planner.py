from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np

from .api import AstarIslandClient
from .baseline import ModelParameters, PredictionPreview, build_all_predictions
from .cache import CacheStore
from .historical import HistoricalSignalPrior, estimate_historical_signal_for_round
from .types import RoundDetail, SimulationResult

PlannerStage = Literal["explore", "infer", "exploit"]


@dataclass(slots=True)
class PlannedQuery:
    stage: str
    seed_index: int
    x: int
    y: int
    w: int
    h: int
    base_score: float
    adjusted_score: float
    entropy_sum: float
    dynamic_sum: float
    unobserved_cells: int
    overlap_ratio: float
    average_support: float


@dataclass(slots=True)
class AutopilotRun:
    requested_queries: int
    executed_queries: int
    batch_stages: list[str]
    batches: list[list[PlannedQuery]]
    results: list[SimulationResult]


@dataclass(frozen=True, slots=True)
class StageProfile:
    name: PlannerStage
    info_weight: float
    entropy_weight: float
    dynamic_weight: float
    unobserved_weight: float
    novelty_weight: float
    overlap_penalty: float
    batch_seed_penalty: float
    seed_balance_boost: float
    focus_boost: float


STAGE_PROFILES: dict[PlannerStage, StageProfile] = {
    "explore": StageProfile(
        name="explore",
        info_weight=0.55,
        entropy_weight=0.12,
        dynamic_weight=0.12,
        unobserved_weight=1.40,
        novelty_weight=0.45,
        overlap_penalty=0.95,
        batch_seed_penalty=0.50,
        seed_balance_boost=0.95,
        focus_boost=0.10,
    ),
    "infer": StageProfile(
        name="infer",
        info_weight=0.85,
        entropy_weight=0.18,
        dynamic_weight=0.22,
        unobserved_weight=0.85,
        novelty_weight=0.28,
        overlap_penalty=0.85,
        batch_seed_penalty=0.25,
        seed_balance_boost=0.45,
        focus_boost=0.20,
    ),
    "exploit": StageProfile(
        name="exploit",
        info_weight=1.15,
        entropy_weight=0.32,
        dynamic_weight=0.36,
        unobserved_weight=0.35,
        novelty_weight=0.15,
        overlap_penalty=0.70,
        batch_seed_penalty=0.08,
        seed_balance_boost=0.10,
        focus_boost=0.60,
    ),
}


def integral_image(grid: np.ndarray) -> np.ndarray:
    return np.pad(np.cumsum(np.cumsum(grid, axis=0), axis=1), ((1, 0), (1, 0)))


def rect_sum(integral: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    x2 = x + w
    y2 = y + h
    return float(integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x])


def score_cells(
    preview: PredictionPreview,
    *,
    entropy_grid: np.ndarray | None = None,
    dynamic_grid: np.ndarray | None = None,
    historical_entropy_grid: np.ndarray | None = None,
    historical_dynamic_grid: np.ndarray | None = None,
    historical_support_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    planning_entropy = preview.entropy_grid if entropy_grid is None else entropy_grid
    planning_dynamic = preview.dynamic_grid if dynamic_grid is None else dynamic_grid
    novelty = 1.0 / np.sqrt(1.0 + preview.support_grid)
    observation_discount = np.where(
        preview.coverage_mask,
        0.35 / (1.0 + preview.sample_count_grid),
        1.0,
    )
    uncertainty = planning_entropy * (0.25 + planning_dynamic)
    cell_scores = uncertainty * (0.55 + (0.45 * novelty)) * observation_discount

    if (
        historical_entropy_grid is not None
        and historical_dynamic_grid is not None
        and historical_support_grid is not None
    ):
        historical_uncertainty = historical_entropy_grid * (0.25 + historical_dynamic_grid)
        historical_weight = np.clip(historical_support_grid / 3.0, 0.0, 0.85)
        historical_scores = historical_uncertainty * (0.75 + (0.25 * novelty)) * observation_discount
        cell_scores = ((1.0 - historical_weight) * cell_scores) + (historical_weight * historical_scores)

    return cell_scores, novelty


def determine_stage(
    *,
    existing_queries: int,
    requested_queries: int,
    executed_queries: int = 0,
) -> PlannerStage:
    horizon = max(1, existing_queries + requested_queries)
    progress = (existing_queries + executed_queries) / horizon
    if progress < 0.30:
        return "explore"
    if progress < 0.75:
        return "infer"
    return "exploit"


def eligible_seeds_for_stage(
    stage: PlannerStage,
    candidate_seeds: list[int],
    seed_query_counts: dict[int, int],
    seed_selected_counts: dict[int, int],
) -> set[int] | None:
    if stage != "explore" or not candidate_seeds:
        return None

    pending_batch_seeds = {seed for seed in candidate_seeds if seed_selected_counts[seed] == 0}
    if pending_batch_seeds:
        min_queries = min(seed_query_counts[seed] for seed in pending_batch_seeds)
        return {seed for seed in pending_batch_seeds if seed_query_counts[seed] == min_queries}

    min_total = min(seed_query_counts[seed] + seed_selected_counts[seed] for seed in candidate_seeds)
    return {
        seed
        for seed in candidate_seeds
        if (seed_query_counts[seed] + seed_selected_counts[seed]) == min_total
    }


def seed_priority_map(predictions: dict[int, PredictionPreview], candidate_seeds: list[int]) -> dict[int, float]:
    masses = {
        seed_index: float(np.sum(predictions[seed_index].entropy_grid * (0.25 + predictions[seed_index].dynamic_grid)))
        for seed_index in candidate_seeds
    }
    if not masses:
        return {}
    max_mass = max(masses.values())
    min_mass = min(masses.values())
    if max_mass <= min_mass:
        return {seed_index: 0.5 for seed_index in candidate_seeds}
    return {
        seed_index: (mass - min_mass) / (max_mass - min_mass)
        for seed_index, mass in masses.items()
    }


def plan_query_batch(
    detail: RoundDetail,
    observations: list[SimulationResult],
    *,
    count: int,
    viewport_w: int = 15,
    viewport_h: int = 15,
    seed_indices: list[int] | None = None,
    params: ModelParameters | None = None,
    historical_prior: HistoricalSignalPrior | None = None,
    stage: PlannerStage = "infer",
) -> list[PlannedQuery]:
    if count <= 0:
        return []

    profile = STAGE_PROFILES[stage]
    candidate_seeds = seed_indices or list(range(detail.seeds_count))
    predictions = build_all_predictions(detail, observations, params=params)
    historical_signals = (
        estimate_historical_signal_for_round(detail, historical_prior)
        if historical_prior is not None
        else {}
    )
    seed_priority = seed_priority_map(predictions, candidate_seeds)

    candidates: list[PlannedQuery] = []
    seed_query_counts = {
        seed_index: sum(1 for observation in observations if observation.seed_index == seed_index)
        for seed_index in candidate_seeds
    }

    for seed_index in candidate_seeds:
        preview = predictions[seed_index]
        planning_entropy = preview.entropy_grid
        planning_dynamic = preview.dynamic_grid
        historical_entropy = None
        historical_dynamic = None
        historical_support = None
        if seed_index in historical_signals:
            historical_entropy, historical_dynamic, historical_support = historical_signals[seed_index]
            blend = np.clip(historical_support / 4.0, 0.0, 0.65)
            planning_entropy = ((1.0 - blend) * planning_entropy) + (blend * historical_entropy)
            planning_dynamic = ((1.0 - blend) * planning_dynamic) + (blend * historical_dynamic)

        cell_scores, novelty = score_cells(
            preview,
            entropy_grid=planning_entropy,
            dynamic_grid=planning_dynamic,
            historical_entropy_grid=historical_entropy,
            historical_dynamic_grid=historical_dynamic,
            historical_support_grid=historical_support,
        )
        score_integral = integral_image(cell_scores)
        entropy_integral = integral_image(planning_entropy)
        dynamic_integral = integral_image(planning_dynamic)
        uncovered_integral = integral_image((~preview.coverage_mask).astype(float))
        novelty_integral = integral_image(novelty)

        max_x = detail.map_width - viewport_w
        max_y = detail.map_height - viewport_h
        area = viewport_w * viewport_h
        for y in range(max_y + 1):
            for x in range(max_x + 1):
                base_score = rect_sum(score_integral, x, y, viewport_w, viewport_h)
                entropy_sum = rect_sum(entropy_integral, x, y, viewport_w, viewport_h)
                dynamic_sum = rect_sum(dynamic_integral, x, y, viewport_w, viewport_h)
                unobserved_cells = int(rect_sum(uncovered_integral, x, y, viewport_w, viewport_h))
                average_support = rect_sum(novelty_integral, x, y, viewport_w, viewport_h) / (viewport_w * viewport_h)

                base_density = base_score / area
                entropy_density = entropy_sum / area
                dynamic_density = dynamic_sum / area
                unobserved_ratio = unobserved_cells / area
                focus_factor = 1.0 + (profile.focus_boost * seed_priority.get(seed_index, 0.0))
                adjusted_score = area * focus_factor * (
                    (profile.info_weight * base_density)
                    + (profile.entropy_weight * entropy_density)
                    + (profile.dynamic_weight * dynamic_density)
                    + (profile.unobserved_weight * unobserved_ratio)
                    + (profile.novelty_weight * average_support)
                )
                candidates.append(
                    PlannedQuery(
                        stage=stage,
                        seed_index=seed_index,
                        x=x,
                        y=y,
                        w=viewport_w,
                        h=viewport_h,
                        base_score=base_score,
                        adjusted_score=adjusted_score,
                        entropy_sum=entropy_sum,
                        dynamic_sum=dynamic_sum,
                        unobserved_cells=unobserved_cells,
                        overlap_ratio=0.0,
                        average_support=average_support,
                    )
                )

    selected: list[PlannedQuery] = []
    planned_masks = {
        seed_index: np.zeros((detail.map_height, detail.map_width), dtype=bool)
        for seed_index in candidate_seeds
    }
    seed_selected_counts = {seed_index: 0 for seed_index in candidate_seeds}

    for _ in range(min(count, len(candidates))):
        best_query: PlannedQuery | None = None
        best_score = -1.0
        eligible_seeds = eligible_seeds_for_stage(stage, candidate_seeds, seed_query_counts, seed_selected_counts)

        for candidate in candidates:
            if any(
                chosen.seed_index == candidate.seed_index
                and chosen.x == candidate.x
                and chosen.y == candidate.y
                and chosen.w == candidate.w
                and chosen.h == candidate.h
                for chosen in selected
            ):
                continue
            if eligible_seeds is not None and candidate.seed_index not in eligible_seeds:
                continue

            mask = planned_masks[candidate.seed_index]
            overlap = mask[candidate.y : candidate.y + candidate.h, candidate.x : candidate.x + candidate.w].mean()
            overlap_penalty = max(0.12, 1.0 - (profile.overlap_penalty * overlap))

            min_existing = min(seed_query_counts[seed] for seed in candidate_seeds)
            existing_gap = seed_query_counts[candidate.seed_index] - min_existing
            balance_boost = 1.0 / (1.0 + (profile.seed_balance_boost * max(0, existing_gap)))

            min_selected = min(seed_selected_counts[seed] for seed in candidate_seeds)
            selected_gap = seed_selected_counts[candidate.seed_index] - min_selected
            batch_penalty = 1.0 / (1.0 + (profile.batch_seed_penalty * max(0, selected_gap)))

            adjusted = candidate.adjusted_score * overlap_penalty * balance_boost * batch_penalty

            if adjusted > best_score:
                best_score = adjusted
                best_query = PlannedQuery(
                    stage=stage,
                    seed_index=candidate.seed_index,
                    x=candidate.x,
                    y=candidate.y,
                    w=candidate.w,
                    h=candidate.h,
                    base_score=candidate.base_score,
                    adjusted_score=adjusted,
                    entropy_sum=candidate.entropy_sum,
                    dynamic_sum=candidate.dynamic_sum,
                    unobserved_cells=candidate.unobserved_cells,
                    overlap_ratio=float(overlap),
                    average_support=candidate.average_support,
                )

        if best_query is None:
            break

        selected.append(best_query)
        planned_masks[best_query.seed_index][
            best_query.y : best_query.y + best_query.h,
            best_query.x : best_query.x + best_query.w,
        ] = True
        seed_selected_counts[best_query.seed_index] += 1

    return selected


def execute_query_plan(
    client: AstarIslandClient,
    cache: CacheStore,
    round_id: str,
    plan: list[PlannedQuery],
    *,
    pause_seconds: float = 0.25,
) -> list[SimulationResult]:
    results: list[SimulationResult] = []

    for index, item in enumerate(plan):
        result = client.simulate(
            round_id=round_id,
            seed_index=item.seed_index,
            viewport_x=item.x,
            viewport_y=item.y,
            viewport_w=item.w,
            viewport_h=item.h,
        )
        cache.append_observation(result)
        results.append(result)
        if index < len(plan) - 1 and pause_seconds > 0:
            time.sleep(pause_seconds)

    return results


def run_iterative_autopilot(
    client: AstarIslandClient,
    cache: CacheStore,
    detail: RoundDetail,
    *,
    round_id: str,
    total_queries: int,
    viewport_w: int = 15,
    viewport_h: int = 15,
    seed_indices: list[int] | None = None,
    replan_every: int = 5,
    pause_seconds: float = 0.25,
    params: ModelParameters | None = None,
    historical_prior: HistoricalSignalPrior | None = None,
) -> AutopilotRun:
    remaining = max(0, total_queries)
    existing_queries = len(cache.load_observations(round_id))
    batch_stages: list[str] = []
    batches: list[list[PlannedQuery]] = []
    results: list[SimulationResult] = []

    while remaining > 0:
        observations = cache.load_observations(round_id)
        stage = determine_stage(
            existing_queries=existing_queries,
            requested_queries=total_queries,
            executed_queries=len(results),
        )
        batch_plan = plan_query_batch(
            detail,
            observations,
            count=min(remaining, max(1, replan_every)),
            viewport_w=viewport_w,
            viewport_h=viewport_h,
            seed_indices=seed_indices,
            params=params,
            historical_prior=historical_prior,
            stage=stage,
        )
        if not batch_plan:
            break

        batch_results = execute_query_plan(
            client,
            cache,
            round_id,
            batch_plan,
            pause_seconds=pause_seconds,
        )
        if not batch_results:
            break

        batch_stages.append(stage)
        batches.append(batch_plan)
        results.extend(batch_results)
        remaining -= len(batch_results)

    return AutopilotRun(
        requested_queries=total_queries,
        executed_queries=len(results),
        batch_stages=batch_stages,
        batches=batches,
        results=results,
    )
