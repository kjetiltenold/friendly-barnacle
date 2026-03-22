from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import BUILDABLE_CODES, CARDINAL_OFFSETS, INTERNAL_TO_CLASS
from .types import InitialState, RoundDetail, SimulationResult

MIN_PROBABILITY_FLOOR = 0.01

FloatGrid = NDArray[np.float64]
IntGrid = NDArray[np.int_]
BoolGrid = NDArray[np.bool_]

SPATIAL_KERNEL: tuple[tuple[int, int, float], ...] = (
    (0, 0, 1.00),
    (-1, 0, 0.55),
    (1, 0, 0.55),
    (0, -1, 0.55),
    (0, 1, 0.55),
    (-1, -1, 0.30),
    (-1, 1, 0.30),
    (1, -1, 0.30),
    (1, 1, 0.30),
    (-2, 0, 0.16),
    (2, 0, 0.16),
    (0, -2, 0.16),
    (0, 2, 0.16),
)


@dataclass(slots=True)
class SeedFeatureMaps:
    initial_grid: IntGrid
    initial_group_grid: IntGrid
    coastal_mask: BoolGrid
    forest_neighbors: IntGrid
    ocean_neighbors: IntGrid
    mountain_neighbors: IntGrid
    nearest_settlement_distance: IntGrid
    nearest_port_distance: IntGrid
    settlement_density: IntGrid
    port_density: IntGrid
    x_zone: IntGrid
    y_zone: IntGrid


@dataclass(slots=True)
class ModelParameters:
    prior_weight: float = 2.5
    historical_weight: float = 1.4
    exact_weight: float = 6.0
    spatial_weight: float = 1.0
    global_scale: float = 1.0
    code_scale: float = 0.9
    frontier_scale: float = 1.1
    context_scale: float = 1.2
    region_scale: float = 0.9


@dataclass(slots=True)
class RoundLatentProfile:
    expansion_bias: float = 0.0
    port_bias: float = 0.0
    collapse_bias: float = 0.0
    forest_bias: float = 0.0
    certainty: float = 0.0


@dataclass(slots=True)
class LearnedTables:
    global_counts: FloatGrid
    code_counts: dict[str, FloatGrid]
    frontier_counts: dict[str, FloatGrid]
    context_counts: dict[str, FloatGrid]
    region_counts: dict[str, FloatGrid]


@dataclass(slots=True)
class PredictionPreview:
    prediction: FloatGrid
    argmax_grid: IntGrid
    confidence_grid: FloatGrid
    coverage_mask: BoolGrid
    entropy_grid: FloatGrid
    dynamic_grid: FloatGrid
    support_grid: FloatGrid
    sample_count_grid: FloatGrid


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


def count_points_within_radius(x: int, y: int, points: list[tuple[int, int]], radius: int) -> int:
    return sum(1 for px, py in points if abs(x - px) + abs(y - py) <= radius)


def bucket_distance(distance: int) -> str:
    if distance < 0:
        return "n"
    if distance == 0:
        return "0"
    if distance <= 2:
        return "1"
    if distance <= 4:
        return "2"
    if distance <= 7:
        return "3"
    return "4"


def bucket_density(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3"


def initial_group_grid(grid: IntGrid) -> IntGrid:
    groups = np.full_like(grid, 3)
    groups[grid == 10] = 0
    groups[grid == 5] = 1
    groups[grid == 4] = 2
    return groups


def build_seed_feature_maps(state: InitialState) -> SeedFeatureMaps:
    grid = overlay_initial_settlements(state)
    height, width = grid.shape
    coastal_mask = np.zeros((height, width), dtype=bool)
    forest_neighbors = np.zeros((height, width), dtype=int)
    ocean_neighbors = np.zeros((height, width), dtype=int)
    mountain_neighbors = np.zeros((height, width), dtype=int)
    nearest_settlement_distance = np.full((height, width), -1, dtype=int)
    nearest_port_distance = np.full((height, width), -1, dtype=int)
    settlement_density = np.zeros((height, width), dtype=int)
    port_density = np.zeros((height, width), dtype=int)
    x_zone = np.zeros((height, width), dtype=int)
    y_zone = np.zeros((height, width), dtype=int)

    settlement_points = [(settlement.x, settlement.y) for settlement in state.settlements if settlement.alive]
    port_points = [(settlement.x, settlement.y) for settlement in state.settlements if settlement.alive and settlement.has_port]

    for y in range(height):
        for x in range(width):
            coastal_mask[y, x] = is_coastal(grid, y, x)
            forest_neighbors[y, x] = count_adjacent(grid, y, x, {4})
            ocean_neighbors[y, x] = count_adjacent(grid, y, x, {10})
            mountain_neighbors[y, x] = count_adjacent(grid, y, x, {5})
            settlement_density[y, x] = count_points_within_radius(x, y, settlement_points, radius=3)
            port_density[y, x] = count_points_within_radius(x, y, port_points, radius=4)

            nearest_settlement = nearest_seed_distance(x, y, settlement_points)
            nearest_port = nearest_seed_distance(x, y, port_points)
            nearest_settlement_distance[y, x] = -1 if nearest_settlement is None else nearest_settlement
            nearest_port_distance[y, x] = -1 if nearest_port is None else nearest_port

            x_zone[y, x] = min(3, (x * 4) // max(1, width))
            y_zone[y, x] = min(3, (y * 4) // max(1, height))

    return SeedFeatureMaps(
        initial_grid=grid,
        initial_group_grid=initial_group_grid(grid),
        coastal_mask=coastal_mask,
        forest_neighbors=forest_neighbors,
        ocean_neighbors=ocean_neighbors,
        mountain_neighbors=mountain_neighbors,
        nearest_settlement_distance=nearest_settlement_distance,
        nearest_port_distance=nearest_port_distance,
        settlement_density=settlement_density,
        port_density=port_density,
        x_zone=x_zone,
        y_zone=y_zone,
    )


def build_feature_maps_for_round(detail: RoundDetail) -> dict[int, SeedFeatureMaps]:
    return {
        seed_index: build_seed_feature_maps(detail.initial_states[seed_index])
        for seed_index in range(detail.seeds_count)
    }


def base_distribution_for_cell(feature_maps: SeedFeatureMaps, y: int, x: int) -> FloatGrid:
    code = int(feature_maps.initial_grid[y, x])
    forest_neighbors = int(feature_maps.forest_neighbors[y, x])
    coastal = bool(feature_maps.coastal_mask[y, x])
    nearest = int(feature_maps.nearest_settlement_distance[y, x])
    influence = 0.0 if nearest < 0 else max(0.0, 1.0 - (nearest / 8.0))

    if code == 10:
        return np.array([0.97, 0.005, 0.005, 0.005, 0.005, 0.01], dtype=float)
    if code == 5:
        return np.array([0.01, 0.005, 0.005, 0.005, 0.005, 0.97], dtype=float)
    if code == 4:
        return np.array([0.05, 0.02, 0.01, 0.03, 0.87, 0.02], dtype=float)
    if code == 1:
        port_bonus = 0.10 if coastal else 0.0
        return np.array([0.12, 0.56, 0.08 + port_bonus, 0.16, 0.04, 0.04], dtype=float)
    if code == 2:
        return np.array([0.08, 0.18, 0.56, 0.11, 0.03, 0.04], dtype=float)
    if code == 3:
        return np.array([0.24, 0.18, 0.06 if coastal else 0.03, 0.30, 0.19, 0.04], dtype=float)

    if code in BUILDABLE_CODES:
        empty = 0.80 - (0.18 * influence)
        settlement = 0.05 + (0.18 * influence) + (0.02 * min(forest_neighbors, 2))
        port = 0.02 + ((0.14 * influence) if coastal else 0.0)
        ruin = 0.03 + (0.08 * influence)
        forest = 0.06 + (0.03 * forest_neighbors)
        mountain = 0.03
        return np.array([empty, settlement, port, ruin, forest, mountain], dtype=float)

    return np.array([0.90, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=float)


def build_prior(state: InitialState) -> FloatGrid:
    feature_maps = build_seed_feature_maps(state)
    height, width = feature_maps.initial_grid.shape
    prior = np.zeros((height, width, 6), dtype=float)

    for y in range(height):
        for x in range(width):
            prior[y, x] = base_distribution_for_cell(feature_maps, y, x)

    return normalize_probabilities(prior)


def clip_factor(value: float, *, minimum: float = 0.4, maximum: float = 1.9) -> float:
    return float(np.clip(value, minimum, maximum))


def adjust_distribution_for_profile(
    distribution: FloatGrid,
    feature_maps: SeedFeatureMaps,
    y: int,
    x: int,
    profile: RoundLatentProfile,
) -> FloatGrid:
    if profile.certainty <= 0.0:
        return distribution

    code = int(feature_maps.initial_grid[y, x])
    coastal = bool(feature_maps.coastal_mask[y, x])

    factors = np.ones(6, dtype=float)
    expansion_factor = clip_factor(1.0 + (0.55 * profile.expansion_bias))
    collapse_factor = clip_factor(1.0 + (0.65 * profile.collapse_bias))
    forest_factor = clip_factor(1.0 + (0.50 * profile.forest_bias))
    port_factor = clip_factor(1.0 + (0.65 * profile.port_bias * (1.0 if coastal else 0.25)))
    empty_factor = clip_factor(
        1.0
        - (0.30 * profile.expansion_bias)
        - (0.25 * profile.collapse_bias)
        - (0.18 * profile.forest_bias),
        minimum=0.35,
        maximum=1.8,
    )

    if code in BUILDABLE_CODES:
        factors[0] *= empty_factor
        factors[1] *= expansion_factor
        factors[2] *= port_factor
        factors[3] *= collapse_factor
        factors[4] *= forest_factor
    elif code == 4:
        factors[4] *= forest_factor
        factors[1] *= clip_factor(1.0 + (0.20 * profile.expansion_bias))
        factors[3] *= clip_factor(1.0 + (0.20 * profile.collapse_bias))
    elif code == 3:
        factors[1] *= clip_factor(1.0 + (0.25 * profile.expansion_bias))
        factors[2] *= clip_factor(1.0 + (0.30 * profile.port_bias * (1.0 if coastal else 0.2)))
        factors[3] *= collapse_factor
        factors[4] *= forest_factor
        factors[0] *= empty_factor
    elif code == 1:
        factors[1] *= expansion_factor
        factors[2] *= clip_factor(1.0 + (0.35 * profile.port_bias * (1.0 if coastal else 0.25)))
        factors[3] *= clip_factor(1.0 + (0.30 * profile.collapse_bias))
    elif code == 2:
        factors[1] *= clip_factor(1.0 + (0.20 * profile.expansion_bias))
        factors[2] *= port_factor
        factors[3] *= clip_factor(1.0 + (0.30 * profile.collapse_bias))

    return normalize_probabilities(distribution * factors)


def infer_round_latent_profile(
    detail: RoundDetail,
    observations: list[SimulationResult],
    feature_maps_by_seed: dict[int, SeedFeatureMaps] | None = None,
) -> RoundLatentProfile:
    if not observations:
        return RoundLatentProfile()

    feature_maps_lookup = feature_maps_by_seed or build_feature_maps_for_round(detail)
    observed_total = np.zeros(6, dtype=float)
    expected_total = np.zeros(6, dtype=float)
    coastal_observed_port = 0.0
    coastal_expected_port = 0.0
    reclaim_observed_forest = 0.0
    reclaim_expected_forest = 0.0
    observed_cells = 0

    for observation in observations:
        feature_maps = feature_maps_lookup[observation.seed_index]
        class_grid = internal_to_class_grid(np.array(observation.grid, dtype=int))
        for local_y in range(observation.viewport.h):
            for local_x in range(observation.viewport.w):
                world_y = observation.viewport.y + local_y
                world_x = observation.viewport.x + local_x
                observed_class = int(class_grid[local_y, local_x])
                observed_total[observed_class] += 1.0
                expected = base_distribution_for_cell(feature_maps, world_y, world_x)
                expected_total += expected
                observed_cells += 1

                if feature_maps.coastal_mask[world_y, world_x]:
                    coastal_observed_port += 1.0 if observed_class == 2 else 0.0
                    coastal_expected_port += float(expected[2])

                if int(feature_maps.initial_grid[world_y, world_x]) in BUILDABLE_CODES | {3}:
                    reclaim_observed_forest += 1.0 if observed_class == 4 else 0.0
                    reclaim_expected_forest += float(expected[4])

    if observed_cells == 0:
        return RoundLatentProfile()

    live_delta = ((observed_total[1] + observed_total[2]) - (expected_total[1] + expected_total[2])) / observed_cells
    ruin_delta = (observed_total[3] - expected_total[3]) / observed_cells
    forest_delta = (reclaim_observed_forest - reclaim_expected_forest) / max(1.0, reclaim_expected_forest + 1.0)
    port_delta = (coastal_observed_port - coastal_expected_port) / max(1.0, coastal_expected_port + 1.0)
    certainty = float(np.clip((observed_cells / 1200.0) + (len(observations) / 25.0), 0.0, 1.0))

    return RoundLatentProfile(
        expansion_bias=float(np.tanh(8.0 * live_delta) * certainty),
        port_bias=float(np.tanh(5.0 * port_delta) * certainty),
        collapse_bias=float(np.tanh(10.0 * ruin_delta) * certainty),
        forest_bias=float(np.tanh(4.0 * forest_delta) * certainty),
        certainty=certainty,
    )


def feature_keys(feature_maps: SeedFeatureMaps, y: int, x: int) -> tuple[str, str, str, str]:
    code = int(feature_maps.initial_grid[y, x])
    coastal = int(feature_maps.coastal_mask[y, x])
    forest = min(int(feature_maps.forest_neighbors[y, x]), 3)
    ocean = min(int(feature_maps.ocean_neighbors[y, x]), 3)
    mountain = min(int(feature_maps.mountain_neighbors[y, x]), 3)
    settlement_distance = bucket_distance(int(feature_maps.nearest_settlement_distance[y, x]))
    port_distance = bucket_distance(int(feature_maps.nearest_port_distance[y, x]))
    settlement_density = bucket_density(int(feature_maps.settlement_density[y, x]))
    port_density = bucket_density(int(feature_maps.port_density[y, x]))
    zone_x = int(feature_maps.x_zone[y, x])
    zone_y = int(feature_maps.y_zone[y, x])

    code_key = f"{code}"
    frontier_key = f"{code}|co{coastal}|ds{settlement_distance}|dp{port_distance}"
    context_key = (
        f"{code}|co{coastal}|fn{forest}|on{ocean}|mn{mountain}|"
        f"sd{settlement_density}|pd{port_density}"
    )
    region_key = f"{code}|zx{zone_x}|zy{zone_y}|ds{settlement_distance}|co{coastal}"
    return code_key, frontier_key, context_key, region_key


def empty_tables() -> LearnedTables:
    return LearnedTables(
        global_counts=np.zeros(6, dtype=float),
        code_counts={},
        frontier_counts={},
        context_counts={},
        region_counts={},
    )


def increment_table(table: dict[str, FloatGrid], key: str, class_index: int) -> None:
    counts = table.setdefault(key, np.zeros(6, dtype=float))
    counts[class_index] += 1.0


def aggregate_observation_counts(
    observations: list[SimulationResult],
    height: int,
    width: int,
) -> tuple[FloatGrid, BoolGrid, FloatGrid]:
    counts = np.zeros((height, width, 6), dtype=float)
    coverage = np.zeros((height, width), dtype=bool)
    sample_counts = np.zeros((height, width), dtype=float)

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

        sample_counts[top:bottom, left:right] += 1.0

    return counts, coverage, sample_counts


def build_feature_model(
    detail: RoundDetail,
    observations: list[SimulationResult],
    feature_maps_by_seed: dict[int, SeedFeatureMaps],
) -> LearnedTables:
    tables = empty_tables()

    for observation in observations:
        feature_maps = feature_maps_by_seed[observation.seed_index]
        class_grid = internal_to_class_grid(np.array(observation.grid, dtype=int))

        for local_y in range(observation.viewport.h):
            for local_x in range(observation.viewport.w):
                world_y = observation.viewport.y + local_y
                world_x = observation.viewport.x + local_x
                class_index = int(class_grid[local_y, local_x])
                code_key, frontier_key, context_key, region_key = feature_keys(feature_maps, world_y, world_x)
                tables.global_counts[class_index] += 1.0
                increment_table(tables.code_counts, code_key, class_index)
                increment_table(tables.frontier_counts, frontier_key, class_index)
                increment_table(tables.context_counts, context_key, class_index)
                increment_table(tables.region_counts, region_key, class_index)

    return tables


def table_distribution(counts: FloatGrid | None, *, cap: float, scale: float) -> tuple[FloatGrid, float]:
    if counts is None:
        return np.zeros(6, dtype=float), 0.0
    total = float(counts.sum())
    if total <= 0.0:
        return np.zeros(6, dtype=float), 0.0
    strength = min(total, cap) * scale
    return (counts / total) * strength, strength


def feature_pseudocounts_for_seed(
    feature_maps: SeedFeatureMaps,
    tables: LearnedTables,
    params: ModelParameters,
) -> tuple[FloatGrid, FloatGrid]:
    height, width = feature_maps.initial_grid.shape
    pseudocounts = np.zeros((height, width, 6), dtype=float)
    support = np.zeros((height, width), dtype=float)

    global_distribution, global_strength = table_distribution(
        tables.global_counts,
        cap=1.5,
        scale=params.global_scale,
    )

    for y in range(height):
        for x in range(width):
            code_key, frontier_key, context_key, region_key = feature_keys(feature_maps, y, x)
            total = np.zeros(6, dtype=float)
            cell_support = 0.0

            total += global_distribution
            cell_support += global_strength

            for table, key, cap, scale in (
                (tables.code_counts, code_key, 1.5, params.code_scale),
                (tables.frontier_counts, frontier_key, 2.5, params.frontier_scale),
                (tables.context_counts, context_key, 3.0, params.context_scale),
                (tables.region_counts, region_key, 2.0, params.region_scale),
            ):
                distribution, strength = table_distribution(table.get(key), cap=cap, scale=scale)
                total += distribution
                cell_support += strength

            pseudocounts[y, x] = total
            support[y, x] = cell_support

    return pseudocounts, support


def spatial_pseudocounts(
    exact_counts: FloatGrid,
    sample_counts: FloatGrid,
    feature_maps: SeedFeatureMaps,
) -> tuple[FloatGrid, FloatGrid]:
    height, width, _ = exact_counts.shape
    blurred = np.zeros_like(exact_counts)
    support = np.zeros((height, width), dtype=float)

    for dy, dx, weight in SPATIAL_KERNEL:
        src_y_start = max(0, -dy)
        src_y_end = min(height, height - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(width, width - dx)

        dst_y_start = max(0, dy)
        dst_y_end = min(height, height + dy)
        dst_x_start = max(0, dx)
        dst_x_end = min(width, width + dx)

        source_groups = feature_maps.initial_group_grid[src_y_start:src_y_end, src_x_start:src_x_end]
        target_groups = feature_maps.initial_group_grid[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
        group_mask = (source_groups == target_groups).astype(float)

        blurred[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += (
            exact_counts[src_y_start:src_y_end, src_x_start:src_x_end] * weight * group_mask[..., None]
        )
        support[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += (
            sample_counts[src_y_start:src_y_end, src_x_start:src_x_end] * weight * group_mask
        )

    normalized = normalize_probabilities(blurred)
    strength = np.clip(support, 0.0, 2.5)
    return normalized * strength[..., None], strength


def build_seed_prediction(
    detail: RoundDetail,
    observations: list[SimulationResult],
    seed_index: int,
    *,
    feature_maps_by_seed: dict[int, SeedFeatureMaps] | None = None,
    tables: LearnedTables | None = None,
    params: ModelParameters | None = None,
    latent_profile: RoundLatentProfile | None = None,
    historical_pseudocounts: tuple[FloatGrid, FloatGrid] | None = None,
) -> PredictionPreview:
    model_params = params or ModelParameters()
    feature_maps_lookup = feature_maps_by_seed or build_feature_maps_for_round(detail)
    feature_maps = feature_maps_lookup[seed_index]
    round_profile = latent_profile or infer_round_latent_profile(detail, observations, feature_maps_lookup)
    prior = normalize_probabilities(
        np.stack(
            [
                np.stack(
                    [
                        adjust_distribution_for_profile(
                            base_distribution_for_cell(feature_maps, y, x),
                            feature_maps,
                            y,
                            x,
                            round_profile,
                        )
                        for x in range(detail.map_width)
                    ],
                    axis=0,
                )
                for y in range(detail.map_height)
            ],
            axis=0,
        )
    )
    exact_counts, coverage, sample_counts = aggregate_observation_counts(observations, detail.map_height, detail.map_width)
    learned_tables = tables or build_feature_model(detail, observations, feature_maps_lookup)
    feature_counts, feature_support = feature_pseudocounts_for_seed(feature_maps, learned_tables, model_params)
    spatial_counts, spatial_support = spatial_pseudocounts(exact_counts, sample_counts, feature_maps)
    if historical_pseudocounts is None:
        historical_counts = np.zeros_like(exact_counts)
        historical_support = np.zeros((detail.map_height, detail.map_width), dtype=float)
    else:
        historical_counts, historical_support = historical_pseudocounts

    posterior = normalize_probabilities(
        (prior * model_params.prior_weight)
        + historical_counts
        + feature_counts
        + (spatial_counts * model_params.spatial_weight)
        + (exact_counts * model_params.exact_weight)
    )
    entropy_grid = -np.sum(posterior * np.log(posterior), axis=-1)
    dynamic_grid = posterior[..., 1] + posterior[..., 2] + posterior[..., 3] + (0.35 * posterior[..., 4])
    argmax_grid = np.argmax(posterior, axis=-1)
    confidence_grid = np.max(posterior, axis=-1)
    support_grid = (
        historical_support
        + feature_support
        + (spatial_support * model_params.spatial_weight)
        + (sample_counts * model_params.exact_weight)
    )

    return PredictionPreview(
        prediction=posterior,
        argmax_grid=argmax_grid,
        confidence_grid=confidence_grid,
        coverage_mask=coverage,
        entropy_grid=entropy_grid,
        dynamic_grid=dynamic_grid,
        support_grid=support_grid,
        sample_count_grid=sample_counts,
    )


def build_all_predictions(
    detail: RoundDetail,
    observations: list[SimulationResult],
    *,
    params: ModelParameters | None = None,
    historical_pseudocounts_by_seed: dict[int, tuple[FloatGrid, FloatGrid]] | None = None,
) -> dict[int, PredictionPreview]:
    model_params = params or ModelParameters()
    feature_maps_by_seed = build_feature_maps_for_round(detail)
    tables = build_feature_model(detail, observations, feature_maps_by_seed)
    latent_profile = infer_round_latent_profile(detail, observations, feature_maps_by_seed)
    predictions: dict[int, PredictionPreview] = {}

    for seed_index in range(detail.seeds_count):
        seed_observations = [item for item in observations if item.seed_index == seed_index]
        predictions[seed_index] = build_seed_prediction(
            detail,
            seed_observations,
            seed_index,
            feature_maps_by_seed=feature_maps_by_seed,
            tables=tables,
            params=model_params,
            latent_profile=latent_profile,
            historical_pseudocounts=None
            if historical_pseudocounts_by_seed is None
            else historical_pseudocounts_by_seed.get(seed_index),
        )

    return predictions
