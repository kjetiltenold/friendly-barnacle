from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import (
    BUILDABLE_CODES,
    CARDINAL_OFFSETS,
    EMPTY_CLASS,
    FOREST_CLASS,
    INTERNAL_TO_CLASS,
    MOUNTAIN_CLASS,
    PORT_CLASS,
    RUIN_CLASS,
    SETTLEMENT_CLASS,
)
from .types import InitialState, RoundDetail, SimulationResult

MIN_PROBABILITY_FLOOR = 0.01

FloatGrid = NDArray[np.float64]
IntGrid = NDArray[np.int_]
BoolGrid = NDArray[np.bool_]


@dataclass(slots=True)
class PredictionPreview:
    prediction: FloatGrid
    argmax_grid: IntGrid
    confidence_grid: FloatGrid
    coverage_mask: BoolGrid


def normalize_probabilities(raw: FloatGrid, *, floor: float = MIN_PROBABILITY_FLOOR) -> FloatGrid:
    class_count = raw.shape[-1]
    if floor * class_count >= 1.0:
        raise ValueError("Probability floor is too large for the number of classes.")

    non_negative = np.maximum(raw, 0.0)
    totals = non_negative.sum(axis=-1, keepdims=True)
    fallback = np.full_like(non_negative, 1.0 / class_count)
    scaled = np.divide(non_negative, totals, out=fallback, where=totals > 0.0)
    remaining_mass = 1.0 - (floor * class_count)
    return (scaled * remaining_mass) + floor


def internal_to_class_grid(grid: IntGrid) -> IntGrid:
    class_grid = np.zeros_like(grid)
    for internal_code, class_index in INTERNAL_TO_CLASS.items():
        class_grid[grid == internal_code] = class_index
    return class_grid


def overlay_initial_settlements(state: InitialState) -> IntGrid:
    grid = np.array(state.grid, dtype=int, copy=True)
    for settlement in state.settlements:
        grid[settlement.y, settlement.x] = 2 if settlement.has_port else 1
    return grid


def is_coastal(grid: IntGrid, y: int, x: int) -> bool:
    for dy, dx in CARDINAL_OFFSETS:
        ny = y + dy
        nx = x + dx
        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and grid[ny, nx] == 10:
            return True
    return False


def count_adjacent(grid: IntGrid, y: int, x: int, targets: set[int]) -> int:
    count = 0
    for dy, dx in CARDINAL_OFFSETS:
        ny = y + dy
        nx = x + dx
        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and int(grid[ny, nx]) in targets:
            count += 1
    return count


def nearest_seed_distance(x: int, y: int, points: list[tuple[int, int]]) -> int | None:
    if not points:
        return None
    return min(abs(x - px) + abs(y - py) for px, py in points)


def base_distribution_for_cell(grid: IntGrid, y: int, x: int, seed_points: list[tuple[int, int]]) -> FloatGrid:
    code = int(grid[y, x])
    forest_neighbors = count_adjacent(grid, y, x, {4})
    coastal = is_coastal(grid, y, x)
    nearest = nearest_seed_distance(x, y, seed_points)
    influence = 0.0 if nearest is None else max(0.0, 1.0 - (nearest / 8.0))

    if code == 10:
        return np.array([0.96, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=float)
    if code == 5:
        return np.array([0.02, 0.01, 0.01, 0.01, 0.01, 0.94], dtype=float)
    if code == 4:
        return np.array([0.05, 0.02, 0.01, 0.03, 0.87, 0.02], dtype=float)
    if code == 1:
        port_bonus = 0.12 if coastal else 0.0
        return np.array([0.14, 0.54, 0.08 + port_bonus, 0.16, 0.04, 0.04], dtype=float)
    if code == 2:
        return np.array([0.10, 0.18, 0.52, 0.14, 0.03, 0.03], dtype=float)
    if code == 3:
        return np.array([0.24, 0.18, 0.08 if coastal else 0.03, 0.28, 0.16, 0.05], dtype=float)

    if code in BUILDABLE_CODES:
        empty = 0.78 - (0.18 * influence)
        settlement = 0.05 + (0.20 * influence) + (0.02 * min(forest_neighbors, 2))
        port = 0.02 + ((0.14 * influence) if coastal else 0.0)
        ruin = 0.03 + (0.09 * influence)
        forest = 0.08 + (0.03 * forest_neighbors)
        mountain = 0.04
        return np.array([empty, settlement, port, ruin, forest, mountain], dtype=float)

    return np.array([0.90, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=float)


def feature_bucket(grid: IntGrid, y: int, x: int, seed_points: list[tuple[int, int]]) -> str:
    code = int(grid[y, x])
    coastal = 1 if is_coastal(grid, y, x) else 0
    forest_neighbors = min(count_adjacent(grid, y, x, {4}), 2)
    nearest = nearest_seed_distance(x, y, seed_points)
    if nearest is None:
        distance_bucket = "none"
    elif nearest == 0:
        distance_bucket = "0"
    elif nearest <= 2:
        distance_bucket = "1"
    elif nearest <= 4:
        distance_bucket = "2"
    elif nearest <= 7:
        distance_bucket = "3"
    else:
        distance_bucket = "4"
    return f"{code}|c{coastal}|f{forest_neighbors}|d{distance_bucket}"


def build_prior(state: InitialState) -> FloatGrid:
    grid = overlay_initial_settlements(state)
    height, width = grid.shape
    prior = np.zeros((height, width, 6), dtype=float)
    seed_points = [(settlement.x, settlement.y) for settlement in state.settlements if settlement.alive]

    for y in range(height):
        for x in range(width):
            prior[y, x] = base_distribution_for_cell(grid, y, x, seed_points)

    return normalize_probabilities(prior)


def aggregate_observation_counts(observations: list[SimulationResult], height: int, width: int) -> tuple[FloatGrid, BoolGrid]:
    counts = np.zeros((height, width, 6), dtype=float)
    coverage = np.zeros((height, width), dtype=bool)

    for observation in observations:
        viewport_grid = np.array(observation.grid, dtype=int)
        class_grid = internal_to_class_grid(viewport_grid)
        top = observation.viewport.y
        left = observation.viewport.x
        bottom = top + observation.viewport.h
        right = left + observation.viewport.w
        coverage[top:bottom, left:right] = True

        for class_index in range(6):
            counts[top:bottom, left:right, class_index] += (class_grid == class_index).astype(float)

    return counts, coverage


def build_feature_model(detail: RoundDetail, observations: list[SimulationResult]) -> dict[str, FloatGrid]:
    bucket_counts: dict[str, FloatGrid] = {}

    for observation in observations:
        state = detail.initial_states[observation.seed_index]
        initial_grid = overlay_initial_settlements(state)
        seed_points = [(settlement.x, settlement.y) for settlement in state.settlements if settlement.alive]
        class_grid = internal_to_class_grid(np.array(observation.grid, dtype=int))

        for local_y in range(observation.viewport.h):
            for local_x in range(observation.viewport.w):
                world_y = observation.viewport.y + local_y
                world_x = observation.viewport.x + local_x
                bucket = feature_bucket(initial_grid, world_y, world_x, seed_points)
                if bucket not in bucket_counts:
                    bucket_counts[bucket] = np.zeros(6, dtype=float)
                observed_class = int(class_grid[local_y, local_x])
                bucket_counts[bucket][observed_class] += 1.0

    return bucket_counts


def feature_pseudocounts_for_seed(
    state: InitialState,
    bucket_counts: dict[str, FloatGrid],
) -> FloatGrid:
    initial_grid = overlay_initial_settlements(state)
    seed_points = [(settlement.x, settlement.y) for settlement in state.settlements if settlement.alive]
    height, width = initial_grid.shape
    counts = np.zeros((height, width, 6), dtype=float)

    for y in range(height):
        for x in range(width):
            bucket = feature_bucket(initial_grid, y, x, seed_points)
            observed = bucket_counts.get(bucket)
            if observed is None:
                continue
            total = float(observed.sum())
            if total <= 0:
                continue
            weight = min(total, 5.0)
            counts[y, x] = (observed / total) * weight

    return counts


def build_seed_prediction(
    detail: RoundDetail,
    observations: list[SimulationResult],
    seed_index: int,
    *,
    bucket_counts: dict[str, FloatGrid] | None = None,
) -> PredictionPreview:
    state = detail.initial_states[seed_index]
    prior = build_prior(state)
    counts, coverage = aggregate_observation_counts(observations, detail.map_height, detail.map_width)
    feature_counts = feature_pseudocounts_for_seed(state, bucket_counts or {})
    posterior = normalize_probabilities((prior * 3.0) + feature_counts + counts)
    argmax_grid = np.argmax(posterior, axis=-1)
    confidence_grid = np.max(posterior, axis=-1)
    return PredictionPreview(
        prediction=posterior,
        argmax_grid=argmax_grid,
        confidence_grid=confidence_grid,
        coverage_mask=coverage,
    )


def build_all_predictions(detail: RoundDetail, observations: list[SimulationResult]) -> dict[int, PredictionPreview]:
    bucket_counts = build_feature_model(detail, observations)
    predictions: dict[int, PredictionPreview] = {}
    for seed_index in range(detail.seeds_count):
        seed_observations = [item for item in observations if item.seed_index == seed_index]
        predictions[seed_index] = build_seed_prediction(
            detail,
            seed_observations,
            seed_index,
            bucket_counts=bucket_counts,
        )
    return predictions
