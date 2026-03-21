from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .baseline import build_feature_maps_for_round, feature_keys
from .types import RoundAnalysis, RoundDetail

MetricStats = tuple[float, float, float]

HISTORICAL_PRIOR_TABLES: tuple[tuple[str, float, float], ...] = (
    ("global", 2.0, 0.35),
    ("code", 6.0, 0.20),
    ("frontier", 8.0, 0.22),
    ("context", 10.0, 0.18),
    ("region", 8.0, 0.16),
)


@dataclass(slots=True)
class HistoricalSignalPrior:
    global_stats: MetricStats = (0.0, 0.0, 0.0)
    code_stats: dict[str, MetricStats] = field(default_factory=dict)
    frontier_stats: dict[str, MetricStats] = field(default_factory=dict)
    context_stats: dict[str, MetricStats] = field(default_factory=dict)
    region_stats: dict[str, MetricStats] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HistoricalSignalPrior":
        def convert_stats(raw: object) -> MetricStats:
            values = raw if isinstance(raw, (list, tuple)) else [0.0, 0.0, 0.0]
            padded = list(values)[:3] + [0.0] * max(0, 3 - len(values))
            return (float(padded[0]), float(padded[1]), float(padded[2]))

        def convert_table(raw: object) -> dict[str, MetricStats]:
            if not isinstance(raw, dict):
                return {}
            return {str(key): convert_stats(value) for key, value in raw.items()}

        return cls(
            global_stats=convert_stats(payload.get("global_stats")),
            code_stats=convert_table(payload.get("code_stats")),
            frontier_stats=convert_table(payload.get("frontier_stats")),
            context_stats=convert_table(payload.get("context_stats")),
            region_stats=convert_table(payload.get("region_stats")),
        )


def _updated_stats(stats: MetricStats | None, entropy: float, dynamic: float) -> MetricStats:
    count, entropy_sum, dynamic_sum = stats or (0.0, 0.0, 0.0)
    return (count + 1.0, entropy_sum + entropy, dynamic_sum + dynamic)


def _ground_truth_entropy(ground_truth: np.ndarray) -> np.ndarray:
    positive = ground_truth > 0.0
    safe = np.where(positive, ground_truth, 1.0)
    return -np.sum(np.where(positive, ground_truth * np.log(safe), 0.0), axis=-1)


def _ground_truth_dynamic(ground_truth: np.ndarray) -> np.ndarray:
    return ground_truth[..., 1] + ground_truth[..., 2] + ground_truth[..., 3] + (0.35 * ground_truth[..., 4])


def build_historical_signal_prior(
    rounds: list[tuple[RoundDetail, dict[int, RoundAnalysis]]],
) -> HistoricalSignalPrior:
    prior = HistoricalSignalPrior()

    for detail, analyses in rounds:
        feature_maps_by_seed = build_feature_maps_for_round(detail)
        for seed_index, analysis in analyses.items():
            feature_maps = feature_maps_by_seed.get(seed_index)
            if feature_maps is None:
                continue
            ground_truth = np.array(analysis.ground_truth, dtype=float)
            entropy_grid = _ground_truth_entropy(ground_truth)
            dynamic_grid = _ground_truth_dynamic(ground_truth)

            for y in range(detail.map_height):
                for x in range(detail.map_width):
                    entropy = float(entropy_grid[y, x])
                    dynamic = float(dynamic_grid[y, x])
                    prior.global_stats = _updated_stats(prior.global_stats, entropy, dynamic)

                    code_key, frontier_key, context_key, region_key = feature_keys(feature_maps, y, x)
                    prior.code_stats[code_key] = _updated_stats(prior.code_stats.get(code_key), entropy, dynamic)
                    prior.frontier_stats[frontier_key] = _updated_stats(
                        prior.frontier_stats.get(frontier_key),
                        entropy,
                        dynamic,
                    )
                    prior.context_stats[context_key] = _updated_stats(
                        prior.context_stats.get(context_key),
                        entropy,
                        dynamic,
                    )
                    prior.region_stats[region_key] = _updated_stats(
                        prior.region_stats.get(region_key),
                        entropy,
                        dynamic,
                    )

    return prior


def _metric_contribution(stats: MetricStats | None, *, cap: float, scale: float) -> tuple[float, float, float]:
    if stats is None:
        return 0.0, 0.0, 0.0
    count, entropy_sum, dynamic_sum = stats
    if count <= 0.0:
        return 0.0, 0.0, 0.0
    strength = min(count, cap) * scale
    return (entropy_sum / count) * strength, (dynamic_sum / count) * strength, strength


def estimate_historical_signal_for_round(
    detail: RoundDetail,
    prior: HistoricalSignalPrior,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    feature_maps_by_seed = build_feature_maps_for_round(detail)
    signals: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for seed_index, feature_maps in feature_maps_by_seed.items():
        entropy_grid = np.zeros((detail.map_height, detail.map_width), dtype=float)
        dynamic_grid = np.zeros((detail.map_height, detail.map_width), dtype=float)
        support_grid = np.zeros((detail.map_height, detail.map_width), dtype=float)

        for y in range(detail.map_height):
            for x in range(detail.map_width):
                code_key, frontier_key, context_key, region_key = feature_keys(feature_maps, y, x)

                entropy_total = 0.0
                dynamic_total = 0.0
                support_total = 0.0

                for table_name, cap, scale in HISTORICAL_PRIOR_TABLES:
                    if table_name == "global":
                        stats = prior.global_stats
                    elif table_name == "code":
                        stats = prior.code_stats.get(code_key)
                    elif table_name == "frontier":
                        stats = prior.frontier_stats.get(frontier_key)
                    elif table_name == "context":
                        stats = prior.context_stats.get(context_key)
                    else:
                        stats = prior.region_stats.get(region_key)

                    entropy_contrib, dynamic_contrib, strength = _metric_contribution(stats, cap=cap, scale=scale)
                    entropy_total += entropy_contrib
                    dynamic_total += dynamic_contrib
                    support_total += strength

                if support_total > 0.0:
                    entropy_grid[y, x] = np.clip(entropy_total / support_total, 0.0, np.log(6.0))
                    dynamic_grid[y, x] = np.clip(dynamic_total / support_total, 0.0, 1.0)
                    support_grid[y, x] = support_total

        signals[seed_index] = (entropy_grid, dynamic_grid, support_grid)

    return signals
