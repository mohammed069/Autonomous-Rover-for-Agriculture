"""Farm environment for the autonomous rover simulation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from config import (
    CROP_RATIO,
    DRY_SOIL_RATIO,
    INITIAL_WEED_RATIO,
    MOISTURE_DECAY,
    OBSTACLE_RATIO,
    OBSTACLE_SPAWN_CHANCE,
    RAIN_MOISTURE_BOOST,
    SUN_MOISTURE_DECAY,
    WEATHER_CHANGE_INTERVAL,
    WEED_SPAWN_CHANCE,
    USELESS_MOVEMENT_PENALTY,
)
from pathfinding import astar
from utils import Action, Cell, CellType, bucket_moisture, clamp, encode_neighborhood, format_position, format_target, in_bounds, manhattan_distance


WEATHER_STATES = ("sunny", "cloudy", "rainy")


@dataclass
class Observation:
    state: tuple
    current_cell: CellType
    moisture_bucket: int
    nearby_weeds: int
    nearby_dry_soil: int
    nearby_obstacles: int


@dataclass
class TargetInfo:
    kind: str
    position: Cell
    path: list[Cell]

    @property
    def distance(self) -> int:
        return max(0, len(self.path) - 1)

    @property
    def description(self) -> str:
        return format_target(self.kind, self.position)


class FarmEnvironment:
    def __init__(self, grid_size: int = 12, seed: int | None = None) -> None:
        self.grid_size = grid_size
        self.seed = seed
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> Cell:
        self.step_count = 0
        self.weather = self.rng.choice(WEATHER_STATES)
        self.weather_timer = 0
        self.grid = np.full((self.grid_size, self.grid_size), CellType.EMPTY, dtype=int)
        self.moisture = np.full((self.grid_size, self.grid_size), 70, dtype=int)

        self._generate_obstacles()
        self._generate_crop_and_soil()
        self._generate_weeds()

        self.rover_start = self._find_free_cell()
        self.initial_task_count = self.remaining_tasks()
        return self.rover_start

    def _generate_obstacles(self) -> None:
        total_cells = self.grid_size * self.grid_size
        obstacle_count = max(1, int(total_cells * OBSTACLE_RATIO))
        for cell in self._random_unique_cells(obstacle_count):
            self.grid[cell] = CellType.OBSTACLE
            self.moisture[cell] = 0

    def _generate_crop_and_soil(self) -> None:
        total_cells = self.grid_size * self.grid_size
        crop_count = max(4, int(total_cells * CROP_RATIO))
        dry_count = max(3, int(total_cells * DRY_SOIL_RATIO))

        for cell in self._random_unique_cells(crop_count):
            if self.grid[cell] != CellType.EMPTY:
                continue
            self.grid[cell] = CellType.CROP
            self.moisture[cell] = self.rng.randint(55, 100)

        for cell in self._random_unique_cells(dry_count):
            if self.grid[cell] != CellType.EMPTY:
                continue
            self.grid[cell] = CellType.DRY_SOIL
            self.moisture[cell] = self.rng.randint(8, 30)

    def _generate_weeds(self) -> None:
        total_cells = self.grid_size * self.grid_size
        weed_count = max(2, int(total_cells * INITIAL_WEED_RATIO))
        candidates = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if self.grid[x, y] in (CellType.EMPTY, CellType.CROP, CellType.WATERED_SOIL)
        ]
        self.rng.shuffle(candidates)
        for cell in candidates[:weed_count]:
            self.grid[cell] = CellType.WEED
            self.moisture[cell] = clamp(int(self.moisture[cell]) + 5, 0, 100)

    def _random_unique_cells(self, count: int) -> List[Cell]:
        cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        self.rng.shuffle(cells)
        return cells[:count]

    def _find_free_cell(self) -> Cell:
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid[x, y] != CellType.OBSTACLE:
                    return (x, y)
        return (0, 0)

    def update_dynamics(self) -> None:
        self.step_count += 1
        self.weather_timer += 1
        if self.weather_timer >= WEATHER_CHANGE_INTERVAL:
            self.weather_timer = 0
            self.weather = self.rng.choice(WEATHER_STATES)

        self._apply_weather()
        self._decay_moisture()
        self._spawn_obstacles()
        self._spawn_weeds()

    def _apply_weather(self) -> None:
        if self.weather == "rainy":
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if self.grid[x, y] != CellType.OBSTACLE:
                        self.moisture[x, y] = clamp(int(self.moisture[x, y]) + RAIN_MOISTURE_BOOST, 0, 100)
        elif self.weather == "sunny":
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if self.grid[x, y] != CellType.OBSTACLE:
                        self.moisture[x, y] = clamp(int(self.moisture[x, y]) - SUN_MOISTURE_DECAY, 0, 100)
        elif self.weather == "cloudy":
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if self.grid[x, y] != CellType.OBSTACLE:
                        self.moisture[x, y] = clamp(int(self.moisture[x, y]) - 1, 0, 100)

    def _decay_moisture(self) -> None:
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid[x, y] == CellType.OBSTACLE:
                    continue
                if self.grid[x, y] == CellType.WATERED_SOIL:
                    self.moisture[x, y] = clamp(int(self.moisture[x, y]) - 1, 0, 100)
                else:
                    self.moisture[x, y] = clamp(int(self.moisture[x, y]) - MOISTURE_DECAY, 0, 100)
                    if self.moisture[x, y] <= 25 and self.grid[x, y] == CellType.CROP:
                        self.grid[x, y] = CellType.DRY_SOIL

    def _spawn_obstacles(self) -> None:
        if self.rng.random() > OBSTACLE_SPAWN_CHANCE:
            return
        candidates = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if self.grid[x, y] in (CellType.EMPTY, CellType.CROP, CellType.DRY_SOIL, CellType.WATERED_SOIL)
        ]
        if not candidates:
            return
        cell = self.rng.choice(candidates)
        self.grid[cell] = CellType.OBSTACLE
        self.moisture[cell] = 0

    def _spawn_weeds(self) -> None:
        if self.rng.random() > WEED_SPAWN_CHANCE:
            return
        candidates = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if self.grid[x, y] in (CellType.EMPTY, CellType.CROP, CellType.DRY_SOIL, CellType.WATERED_SOIL)
        ]
        if not candidates:
            return
        cell = self.rng.choice(candidates)
        self.grid[cell] = CellType.WEED

    def observe(self, rover_position: Cell) -> Observation:
        x, y = rover_position
        neighborhood = self._get_neighborhood(rover_position)
        neighborhood_encoded = encode_neighborhood(neighborhood)
        current_cell = CellType(self.grid[x, y])
        moisture_bucket = bucket_moisture(int(self.moisture[x, y]))
        nearby_weeds = sum(1 for row in neighborhood for value in row if CellType(value) == CellType.WEED)
        nearby_dry_soil = sum(1 for row in neighborhood for value in row if CellType(value) == CellType.DRY_SOIL)
        nearby_obstacles = sum(1 for row in neighborhood for value in row if CellType(value) == CellType.OBSTACLE)
        state = (x, y, int(current_cell), moisture_bucket, nearby_weeds, nearby_dry_soil, nearby_obstacles) + neighborhood_encoded
        return Observation(state, current_cell, moisture_bucket, nearby_weeds, nearby_dry_soil, nearby_obstacles)

    def _get_neighborhood(self, rover_position: Cell):
        x, y = rover_position
        cells = []
        for dx in (-1, 0, 1):
            row = []
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if in_bounds((nx, ny), self.grid_size):
                    row.append(int(self.grid[nx, ny]))
                else:
                    row.append(int(CellType.OBSTACLE))
            cells.append(row)
        return cells

    def walkable(self, position: Cell) -> bool:
        return in_bounds(position, self.grid_size) and CellType(self.grid[position]) != CellType.OBSTACLE

    def reward_for_task_completion(self, before_remaining: int, after_remaining: int) -> float:
        return 0.0

    def remaining_tasks(self) -> int:
        weeds = int(np.sum(self.grid == CellType.WEED))
        dry = int(np.sum(self.grid == CellType.DRY_SOIL))
        return weeds + dry

    def all_targets(self) -> list[TargetInfo]:
        targets: list[TargetInfo] = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell_type = CellType(self.grid[x, y])
                if cell_type in (CellType.WEED, CellType.DRY_SOIL):
                    kind = "Weed" if cell_type == CellType.WEED else "Dry Soil"
                    targets.append(TargetInfo(kind=kind, position=(x, y), path=[]))
        return targets

    def _reachable_targets(self, start: Cell, kinds: Iterable[str] | None = None) -> list[TargetInfo]:
        reachable: list[TargetInfo] = []
        allowed = {kind.lower() for kind in kinds} if kinds is not None else None
        for target in self.all_targets():
            if allowed is not None and target.kind.lower() not in allowed:
                continue
            path = astar(start, target.position, self.grid)
            if path:
                reachable.append(TargetInfo(kind=target.kind, position=target.position, path=path))
        reachable.sort(key=lambda item: (item.distance + (0.0 if item.kind == "Weed" else 0.75), item.distance))
        return reachable

    def scan_targets(self, start: Cell) -> tuple[Optional[TargetInfo], Optional[TargetInfo], Optional[TargetInfo]]:
        weed_targets = self._reachable_targets(start, kinds=("Weed",))
        dry_targets = self._reachable_targets(start, kinds=("Dry Soil",))
        all_targets = self._reachable_targets(start)
        nearest_weed = weed_targets[0] if weed_targets else None
        nearest_dry = dry_targets[0] if dry_targets else None
        preferred = all_targets[0] if all_targets else None
        return nearest_weed, nearest_dry, preferred

    def nearest_target(self, start: Cell) -> Optional[TargetInfo]:
        _, _, preferred = self.scan_targets(start)
        return preferred

    def execute_task(self, position: Cell, action: Action) -> tuple[float, str, bool, str]:
        cell_type = CellType(self.grid[position])
        if action == Action.WATER:
            if cell_type == CellType.DRY_SOIL:
                self.grid[position] = CellType.WATERED_SOIL
                self.moisture[position] = 100
                return 10.0, "watered dry soil", True, "Dry Soil"
            return -1.0, "watering failed", False, ""

        if action == Action.REMOVE_WEED:
            if cell_type == CellType.WEED:
                self.grid[position] = CellType.EMPTY
                self.moisture[position] = clamp(int(self.moisture[position]) + 5, 0, 100)
                return 15.0, "weed removed", True, "Weed"
            return -1.0, "removal failed", False, ""

        if action == Action.IDLE:
            return -2.0, "idle", False, ""

        return -1.0, "noop", False, ""

    def movement_reward(self, previous_distance: int, next_distance: int) -> float:
        if next_distance < previous_distance:
            return 3.0
        return USELESS_MOVEMENT_PENALTY

    def collision_penalty(self) -> float:
        return -10.0

    def target_description(self, target: Optional[TargetInfo]) -> str:
        return target.description if target is not None else "None"

    def describe_cell(self, position: Cell) -> str:
        cell_type = CellType(self.grid[position])
        if cell_type == CellType.WEED:
            return "Weed"
        if cell_type == CellType.DRY_SOIL:
            return "Dry Soil"
        if cell_type == CellType.OBSTACLE:
            return "Obstacle"
        if cell_type == CellType.WATERED_SOIL:
            return "Watered Soil"
        if cell_type == CellType.CROP:
            return "Crop"
        return "Empty"
