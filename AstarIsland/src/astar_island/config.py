from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .constants import BASE_URL


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    env_path = path or Path(".env")
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key and key not in os.environ:
            values[key] = value
    return values


@dataclass(slots=True)
class AppConfig:
    access_token: str | None
    base_url: str
    data_dir: Path
    request_timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "AppConfig":
        dotenv_values = load_dotenv(env_path)
        access_token = os.getenv("ASTAR_ISLAND_ACCESS_TOKEN", dotenv_values.get("ASTAR_ISLAND_ACCESS_TOKEN"))
        base_url = os.getenv("ASTAR_ISLAND_BASE_URL", dotenv_values.get("ASTAR_ISLAND_BASE_URL", BASE_URL))
        data_dir_raw = os.getenv("ASTAR_ISLAND_DATA_DIR", dotenv_values.get("ASTAR_ISLAND_DATA_DIR", ".data"))
        return cls(
            access_token=access_token,
            base_url=base_url.rstrip("/"),
            data_dir=Path(data_dir_raw),
        )
