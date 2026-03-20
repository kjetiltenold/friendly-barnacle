from __future__ import annotations

from typing import Final

BASE_URL: Final[str] = "https://api.ainm.no/astar-island"

EMPTY_CLASS: Final[int] = 0
SETTLEMENT_CLASS: Final[int] = 1
PORT_CLASS: Final[int] = 2
RUIN_CLASS: Final[int] = 3
FOREST_CLASS: Final[int] = 4
MOUNTAIN_CLASS: Final[int] = 5

CLASS_NAMES: Final[tuple[str, ...]] = (
    "Empty",
    "Settlement",
    "Port",
    "Ruin",
    "Forest",
    "Mountain",
)

INTERNAL_TO_CLASS: Final[dict[int, int]] = {
    10: EMPTY_CLASS,
    11: EMPTY_CLASS,
    0: EMPTY_CLASS,
    1: SETTLEMENT_CLASS,
    2: PORT_CLASS,
    3: RUIN_CLASS,
    4: FOREST_CLASS,
    5: MOUNTAIN_CLASS,
}

BUILDABLE_CODES: Final[set[int]] = {0, 1, 2, 3, 11}
LAND_CODES: Final[set[int]] = {0, 1, 2, 3, 4, 5, 11}
STATIC_CODES: Final[set[int]] = {4, 5, 10, 11, 0}

CARDINAL_OFFSETS: Final[tuple[tuple[int, int], ...]] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)

INTERNAL_PALETTE: Final[dict[int, tuple[int, int, int]]] = {
    10: (32, 80, 153),
    11: (198, 183, 126),
    0: (227, 218, 189),
    1: (184, 93, 57),
    2: (66, 133, 204),
    3: (125, 103, 88),
    4: (68, 120, 72),
    5: (102, 105, 110),
}

CLASS_PALETTE: Final[dict[int, tuple[int, int, int]]] = {
    0: (220, 211, 183),
    1: (184, 93, 57),
    2: (66, 133, 204),
    3: (125, 103, 88),
    4: (68, 120, 72),
    5: (102, 105, 110),
}
