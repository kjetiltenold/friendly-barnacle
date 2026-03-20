from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Settlement:
    x: int
    y: int
    has_port: bool
    alive: bool
    population: float | None = None
    food: float | None = None
    wealth: float | None = None
    defense: float | None = None
    owner_id: int | None = None

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "Settlement":
        return cls(
            x=payload["x"],
            y=payload["y"],
            has_port=payload.get("has_port", False),
            alive=payload.get("alive", True),
            population=payload.get("population"),
            food=payload.get("food"),
            wealth=payload.get("wealth"),
            defense=payload.get("defense"),
            owner_id=payload.get("owner_id"),
        )


@dataclass(slots=True)
class InitialState:
    grid: list[list[int]]
    settlements: list[Settlement]

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "InitialState":
        settlements = [Settlement.from_api(item) for item in payload.get("settlements", [])]
        return cls(grid=payload["grid"], settlements=settlements)


@dataclass(slots=True)
class RoundSummary:
    id: str
    round_number: int
    status: str
    map_width: int
    map_height: int
    event_date: str | None = None
    prediction_window_minutes: int | None = None
    started_at: str | None = None
    closes_at: str | None = None
    round_weight: float | None = None
    created_at: str | None = None
    seeds_count: int | None = None

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "RoundSummary":
        return cls(
            id=payload["id"],
            round_number=payload["round_number"],
            status=payload["status"],
            map_width=payload["map_width"],
            map_height=payload["map_height"],
            event_date=payload.get("event_date"),
            prediction_window_minutes=payload.get("prediction_window_minutes"),
            started_at=payload.get("started_at"),
            closes_at=payload.get("closes_at"),
            round_weight=payload.get("round_weight"),
            created_at=payload.get("created_at"),
            seeds_count=payload.get("seeds_count"),
        )


@dataclass(slots=True)
class TeamRoundSummary:
    id: str
    round_number: int
    status: str
    map_width: int
    map_height: int
    seeds_count: int
    round_weight: float | None = None
    round_score: float | None = None
    seed_scores: list[float] | None = None
    seeds_submitted: int | None = None
    rank: int | None = None
    total_teams: int | None = None
    queries_used: int | None = None
    queries_max: int | None = None
    initial_grid: list[list[int]] | None = None
    event_date: str | None = None
    started_at: str | None = None
    closes_at: str | None = None

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "TeamRoundSummary":
        return cls(
            id=payload["id"],
            round_number=payload["round_number"],
            status=payload["status"],
            map_width=payload["map_width"],
            map_height=payload["map_height"],
            seeds_count=payload["seeds_count"],
            round_weight=payload.get("round_weight"),
            round_score=payload.get("round_score"),
            seed_scores=payload.get("seed_scores"),
            seeds_submitted=payload.get("seeds_submitted"),
            rank=payload.get("rank"),
            total_teams=payload.get("total_teams"),
            queries_used=payload.get("queries_used"),
            queries_max=payload.get("queries_max"),
            initial_grid=payload.get("initial_grid"),
            event_date=payload.get("event_date"),
            started_at=payload.get("started_at"),
            closes_at=payload.get("closes_at"),
        )


@dataclass(slots=True)
class RoundDetail:
    id: str
    round_number: int
    status: str
    map_width: int
    map_height: int
    seeds_count: int
    initial_states: list[InitialState]

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "RoundDetail":
        states = [InitialState.from_api(item) for item in payload.get("initial_states", [])]
        return cls(
            id=payload["id"],
            round_number=payload["round_number"],
            status=payload["status"],
            map_width=payload["map_width"],
            map_height=payload["map_height"],
            seeds_count=payload["seeds_count"],
            initial_states=states,
        )


@dataclass(slots=True)
class Budget:
    round_id: str
    queries_used: int
    queries_max: int
    active: bool

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "Budget":
        return cls(
            round_id=payload["round_id"],
            queries_used=payload["queries_used"],
            queries_max=payload["queries_max"],
            active=payload["active"],
        )


@dataclass(slots=True)
class Viewport:
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "Viewport":
        return cls(x=payload["x"], y=payload["y"], w=payload["w"], h=payload["h"])


@dataclass(slots=True)
class SimulationResult:
    round_id: str
    seed_index: int
    grid: list[list[int]]
    settlements: list[Settlement]
    viewport: Viewport
    width: int
    height: int
    queries_used: int
    queries_max: int
    captured_at: str | None = None

    @classmethod
    def from_api(
        cls,
        payload: dict[str, Any],
        *,
        round_id: str,
        seed_index: int,
        captured_at: str | None = None,
    ) -> "SimulationResult":
        settlements = [Settlement.from_api(item) for item in payload.get("settlements", [])]
        return cls(
            round_id=round_id,
            seed_index=seed_index,
            grid=payload["grid"],
            settlements=settlements,
            viewport=Viewport.from_api(payload["viewport"]),
            width=payload["width"],
            height=payload["height"],
            queries_used=payload["queries_used"],
            queries_max=payload["queries_max"],
            captured_at=captured_at,
        )


@dataclass(slots=True)
class RoundAnalysis:
    prediction: list[list[list[float]]]
    ground_truth: list[list[list[float]]]
    score: float | None
    width: int
    height: int
    initial_grid: list[list[int]] | None = None

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "RoundAnalysis":
        return cls(
            prediction=payload["prediction"],
            ground_truth=payload["ground_truth"],
            score=payload.get("score"),
            width=payload["width"],
            height=payload["height"],
            initial_grid=payload.get("initial_grid"),
        )
