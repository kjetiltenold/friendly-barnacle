from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .baseline import ModelParameters
from .types import RoundAnalysis, RoundDetail, SimulationResult


class CacheStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _round_dir(self, round_id: str) -> Path:
        path = self.root / round_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _predictions_dir(self, round_id: str) -> Path:
        path = self._round_dir(round_id) / "predictions"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_round_detail(self, detail: RoundDetail) -> Path:
        path = self._round_dir(detail.id) / "round.json"
        path.write_text(json.dumps(asdict(detail), indent=2))
        return path

    def load_round_detail(self, round_id: str) -> RoundDetail | None:
        path = self._round_dir(round_id) / "round.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        return RoundDetail.from_api(payload)

    def append_observation(self, observation: SimulationResult) -> Path:
        path = self._round_dir(observation.round_id) / "observations.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(observation)))
            handle.write("\n")
        return path

    def load_observations(self, round_id: str, *, seed_index: int | None = None) -> list[SimulationResult]:
        path = self._round_dir(round_id) / "observations.jsonl"
        if not path.exists():
            return []

        results: list[SimulationResult] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                result = SimulationResult.from_api(
                    payload,
                    round_id=payload["round_id"],
                    seed_index=payload["seed_index"],
                    captured_at=payload.get("captured_at"),
                )
                if seed_index is None or result.seed_index == seed_index:
                    results.append(result)
        return results

    def save_prediction(self, round_id: str, seed_index: int, prediction: list[list[list[float]]]) -> Path:
        path = self._predictions_dir(round_id) / f"seed_{seed_index}.json"
        payload: dict[str, Any] = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        }
        path.write_text(json.dumps(payload))
        return path

    def load_prediction(self, round_id: str, seed_index: int) -> list[list[list[float]]] | None:
        path = self._predictions_dir(round_id) / f"seed_{seed_index}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        return payload["prediction"]

    def save_round_analysis(self, round_id: str, seed_index: int, analysis: RoundAnalysis) -> Path:
        path = self._round_dir(round_id) / f"analysis_seed_{seed_index}.json"
        path.write_text(json.dumps(asdict(analysis)))
        return path

    def load_round_analysis(self, round_id: str, seed_index: int) -> RoundAnalysis | None:
        path = self._round_dir(round_id) / f"analysis_seed_{seed_index}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        return RoundAnalysis.from_api(payload)

    def load_all_round_analyses(self, round_id: str) -> dict[int, RoundAnalysis]:
        analyses: dict[int, RoundAnalysis] = {}
        round_dir = self._round_dir(round_id)
        for path in sorted(round_dir.glob("analysis_seed_*.json")):
            seed_index = int(path.stem.split("_")[-1])
            analyses[seed_index] = RoundAnalysis.from_api(json.loads(path.read_text()))
        return analyses

    def save_model_parameters(self, params: ModelParameters) -> Path:
        path = self.root / "model_params.json"
        path.write_text(json.dumps(asdict(params), indent=2))
        return path

    def load_model_parameters(self) -> ModelParameters | None:
        path = self.root / "model_params.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        return ModelParameters(**payload)

    def save_report(self, name: str, payload: dict[str, Any]) -> Path:
        reports_dir = self.root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2))
        return path
