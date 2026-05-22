"""Utility helpers and shared enums for the rover project."""

from __future__ import annotations

import random
from enum import IntEnum
from typing import Sequence, Tuple

import numpy as np


class CellType(IntEnum):
    EMPTY = 0
    CROP = 1
    WEED = 2
    DRY_SOIL = 3
    OBSTACLE = 4
    WATERED_SOIL = 5


class Action(IntEnum):
    MOVE = 0
    WATER = 1
    REMOVE_WEED = 2
    IDLE = 3


Cell = Tuple[int, int]


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def manhattan_distance(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(position: Cell, grid_size: int) -> bool:
    return 0 <= position[0] < grid_size and 0 <= position[1] < grid_size


def neighbors4(position: Cell, grid_size: int):
    x, y = position
    candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    return [candidate for candidate in candidates if in_bounds(candidate, grid_size)]


def action_name(action: Action | int) -> str:
    action = Action(action)
    return {
        Action.MOVE: "MOVE",
        Action.WATER: "WATER",
        Action.REMOVE_WEED: "REMOVE_WEED",
        Action.IDLE: "IDLE",
    }[action]


def format_position(position: Cell) -> str:
    return f"({position[0]},{position[1]})"


def format_target(kind: str, position: Cell) -> str:
    return f"{kind} at {format_position(position)}"


def cell_label(cell_type: CellType | int) -> str:
    cell_type = CellType(cell_type)
    return {
        CellType.EMPTY: "EMPTY",
        CellType.CROP: "CROP",
        CellType.WEED: "WEED",
        CellType.DRY_SOIL: "DRY_SOIL",
        CellType.OBSTACLE: "OBSTACLE",
        CellType.WATERED_SOIL: "WATERED_SOIL",
    }[cell_type]


def bucket_moisture(moisture: int) -> int:
    if moisture < 25:
        return 0
    if moisture < 50:
        return 1
    if moisture < 75:
        return 2
    return 3


def encode_neighborhood(neighborhood: Sequence[Sequence[int]]) -> Tuple[int, ...]:
    encoded: list[int] = []
    for row in neighborhood:
        encoded.extend(int(value) for value in row)
    return tuple(encoded)


def describe_status(cell_type: CellType, moisture_bucket: int, weed_count: int, obstacle_count: int) -> str:
    if cell_type == CellType.OBSTACLE:
        return "Blocked"
    if weed_count > 0:
        return "Weed nearby"
    if cell_type == CellType.DRY_SOIL or moisture_bucket == 0:
        return "Dry soil"
    if obstacle_count > 0:
        return "Obstacle nearby"
    return "Clear"
