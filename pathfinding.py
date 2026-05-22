"""A* pathfinding for navigating the farm field."""

from __future__ import annotations

import heapq
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from utils import Cell, CellType, manhattan_distance, neighbors4


def astar(start: Cell, goal: Cell, grid: Sequence[Sequence[int]]) -> List[Cell]:
    if start == goal:
        return [start]

    grid_size = len(grid)
    open_heap: list[tuple[int, int, Cell]] = []
    heapq.heappush(open_heap, (0, 0, start))
    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, int] = {start: 0}
    visited: set[Cell] = set()

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return _reconstruct_path(came_from, current)

        for neighbor in neighbors4(current, grid_size):
            if CellType(grid[neighbor[0]][neighbor[1]]) == CellType.OBSTACLE:
                continue

            tentative_g = current_cost + 1
            if tentative_g < g_score.get(neighbor, 10**9):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                priority = tentative_g + manhattan_distance(neighbor, goal)
                heapq.heappush(open_heap, (priority, tentative_g, neighbor))

    return []


def _reconstruct_path(came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def nearest_reachable_target(start: Cell, targets: Iterable[Cell], grid: Sequence[Sequence[int]]) -> tuple[Optional[Cell], List[Cell]]:
    best_target: Optional[Cell] = None
    best_path: List[Cell] = []

    for target in targets:
        path = astar(start, target, grid)
        if not path:
            continue
        if not best_path or len(path) < len(best_path):
            best_target = target
            best_path = path

    return best_target, best_path
