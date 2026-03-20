from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .api import AstarIslandClient
from .baseline import PredictionPreview, build_all_predictions
from .cache import CacheStore
from .types import RoundDetail, SimulationResult


@dataclass(slots=True)
class PlannedQuery:
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
    batches: list[list[PlannedQuery]]
    results: list[SimulationResult]


def integral_image(grid: np.ndarray) -> np.ndarray:
    return np.pad(np.cumsum(np.cumsum(grid, axis=0), axis=1), ((1, 0), (1, 0)))


def rect_sum(integral: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    x2 = x + w
    y2 = y + h
    return float(integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x])


def score_cells(preview: PredictionPreview) -> tuple[np.ndarray, np.ndarray]:
    novelty = 1.0 / np.sqrt(1.0 + preview.support_grid)
    observation_discount = np.where(
        preview.coverage_mask,
        0.35 / (1.0 + preview.sample_count_grid),
        1.0,
    )
    uncertainty = preview.entropy_grid * (0.25 + preview.dynamic_grid)
    cell_scores = uncertainty * (0.55 + (0.45 * novelty)) * observation_discount
    return cell_scores, novelty


def plan_query_batch(
    detail: RoundDetail,
    observations: list[SimulationResult],
    *,
    count: int,
    viewport_w: int = 15,
    viewport_h: int = 15,
    seed_indices: list[int] | None = None,
) -> list[PlannedQuery]:
    if count <= 0:
        return []

    candidate_seeds = seed_indices or list(range(detail.seeds_count))
    predictions = build_all_predictions(detail, observations)

    candidates: list[PlannedQuery] = []
    seed_query_counts = {
        seed_index: sum(1 for observation in observations if observation.seed_index == seed_index)
        for seed_index in candidate_seeds
    }

    for seed_index in candidate_seeds:
        preview = predictions[seed_index]
        cell_scores, novelty = score_cells(preview)
        score_integral = integral_image(cell_scores)
        entropy_integral = integral_image(preview.entropy_grid)
        dynamic_integral = integral_image(preview.dynamic_grid)
        uncovered_integral = integral_image((~preview.coverage_mask).astype(float))
        novelty_integral = integral_image(novelty)

        max_x = detail.map_width - viewport_w
        max_y = detail.map_height - viewport_h
        for y in range(max_y + 1):
            for x in range(max_x + 1):
                base_score = rect_sum(score_integral, x, y, viewport_w, viewport_h)
                entropy_sum = rect_sum(entropy_integral, x, y, viewport_w, viewport_h)
                dynamic_sum = rect_sum(dynamic_integral, x, y, viewport_w, viewport_h)
                unobserved_cells = int(rect_sum(uncovered_integral, x, y, viewport_w, viewport_h))
                average_support = rect_sum(novelty_integral, x, y, viewport_w, viewport_h) / (viewport_w * viewport_h)

                seed_penalty = 1.0 / (1.0 + (0.12 * seed_query_counts[seed_index]))
                adjusted_score = base_score * seed_penalty
                candidates.append(
                    PlannedQuery(
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

            mask = planned_masks[candidate.seed_index]
            overlap = mask[candidate.y : candidate.y + candidate.h, candidate.x : candidate.x + candidate.w].mean()
            overlap_penalty = max(0.15, 1.0 - (0.85 * overlap))
            seed_penalty = 1.0 / (1.0 + (0.20 * seed_selected_counts[candidate.seed_index]))
            adjusted = candidate.adjusted_score * overlap_penalty * seed_penalty

            if adjusted > best_score:
                best_score = adjusted
                best_query = PlannedQuery(
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
) -> AutopilotRun:
    remaining = max(0, total_queries)
    batches: list[list[PlannedQuery]] = []
    results: list[SimulationResult] = []

    while remaining > 0:
        observations = cache.load_observations(round_id)
        batch_plan = plan_query_batch(
            detail,
            observations,
            count=min(remaining, max(1, replan_every)),
            viewport_w=viewport_w,
            viewport_h=viewport_h,
            seed_indices=seed_indices,
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

        batches.append(batch_plan)
        results.extend(batch_results)
        remaining -= len(batch_results)

    return AutopilotRun(
        requested_queries=total_queries,
        executed_queries=len(results),
        batches=batches,
        results=results,
    )
