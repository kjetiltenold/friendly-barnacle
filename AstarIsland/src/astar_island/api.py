from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import requests

from .config import AppConfig
from .types import Budget, RoundAnalysis, RoundDetail, RoundSummary, SimulationResult, TeamRoundSummary


class AstarIslandApiError(RuntimeError):
    """Raised when the API returns an error response."""


class AstarIslandClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        if config.access_token:
            self.session.headers["Authorization"] = f"Bearer {config.access_token}"

    def _url(self, path: str) -> str:
        return f"{self.config.base_url}/{path.lstrip('/')}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        auth_required: bool = False,
    ) -> Any:
        if auth_required and "Authorization" not in self.session.headers:
            raise AstarIslandApiError(
                "This endpoint requires authentication. Set ASTAR_ISLAND_ACCESS_TOKEN first."
            )

        url = self._url(path)
        try:
            response = self.session.request(
                method,
                url,
                json=json_payload,
                timeout=self.config.request_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise AstarIslandApiError(f"{method} {url} failed: {exc}") from exc

        if not response.ok:
            message = response.text.strip() or response.reason
            raise AstarIslandApiError(f"{method} {url} failed with {response.status_code}: {message}")

        if response.status_code == 204:
            return None
        return response.json()

    def get_rounds(self) -> list[RoundSummary]:
        payload = self._request("GET", "rounds")
        return [RoundSummary.from_api(item) for item in payload]

    def get_active_round(self) -> RoundSummary | None:
        return next((round_item for round_item in self.get_rounds() if round_item.status == "active"), None)

    def get_round_detail(self, round_id: str) -> RoundDetail:
        payload = self._request("GET", f"rounds/{round_id}")
        return RoundDetail.from_api(payload)

    def get_budget(self) -> Budget:
        payload = self._request("GET", "budget", auth_required=True)
        return Budget.from_api(payload)

    def get_my_rounds(self) -> list[TeamRoundSummary]:
        payload = self._request("GET", "my-rounds", auth_required=True)
        return [TeamRoundSummary.from_api(item) for item in payload]

    def simulate(
        self,
        *,
        round_id: str,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int,
        viewport_h: int,
    ) -> SimulationResult:
        payload = self._request(
            "POST",
            "simulate",
            auth_required=True,
            json_payload={
                "round_id": round_id,
                "seed_index": seed_index,
                "viewport_x": viewport_x,
                "viewport_y": viewport_y,
                "viewport_w": viewport_w,
                "viewport_h": viewport_h,
            },
        )
        return SimulationResult.from_api(
            payload,
            round_id=round_id,
            seed_index=seed_index,
            captured_at=datetime.now(UTC).isoformat(),
        )

    def submit_prediction(self, *, round_id: str, seed_index: int, prediction: list[list[list[float]]]) -> dict[str, Any]:
        return self._request(
            "POST",
            "submit",
            auth_required=True,
            json_payload={
                "round_id": round_id,
                "seed_index": seed_index,
                "prediction": prediction,
            },
        )

    def get_round_analysis(self, *, round_id: str, seed_index: int) -> RoundAnalysis:
        payload = self._request(
            "GET",
            f"analysis/{round_id}/{seed_index}",
            auth_required=True,
        )
        return RoundAnalysis.from_api(payload)
