"""Autonomous rover agent and control logic."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from environment import FarmEnvironment, TargetInfo
from qlearning import QLearningAgent
from utils import Action, Cell, CellType, action_name, manhattan_distance


@dataclass
class RoverStepResult:
    reward: float
    learning_reward: float
    action: Action
    actual_action: str
    target: str
    target_position: Optional[Cell]
    path_length: int
    status: str
    done: bool


@dataclass
class Rover:
    position: Cell
    total_reward: float = 0.0
    episode_reward: float = 0.0
    tasks_completed: int = 0
    weeds_removed: int = 0
    cells_watered: int = 0
    collisions: int = 0
    avoided_collisions: int = 0
    idle_actions: int = 0
    movement_actions: int = 0
    useless_moves: int = 0
    current_path: List[Cell] = field(default_factory=list)
    current_target: Optional[Cell] = None
    last_action: Action = Action.IDLE
    heading: Cell = (0, 1)
    recent_positions: Deque[Cell] = field(default_factory=lambda: deque(maxlen=8))
    step_count: int = 0
    total_steps: int = 0

    def reset(self, position: Cell) -> None:
        self.position = position
        self.episode_reward = 0.0
        self.step_count = 0
        self.current_path = []
        self.current_target = None
        self.last_action = Action.IDLE
        self.heading = (0, 1)
        self.recent_positions.clear()
        self.recent_positions.append(position)

    def step(self, env: FarmEnvironment, agent: QLearningAgent, training: bool = True) -> tuple[tuple, tuple, RoverStepResult]:
        observation = env.observe(self.position)
        state = observation.state
        nearest_weed, nearest_dry, preferred_target = env.scan_targets(self.position)

        if preferred_target is None:
            chosen_action = Action.IDLE
            preferred_action = Action.IDLE
        elif preferred_target.position == self.position:
            preferred_action = Action.REMOVE_WEED if preferred_target.kind == "Weed" else Action.WATER
            chosen_action = preferred_action
        else:
            preferred_action = Action.MOVE
            chosen_action = agent.select_action(
                state,
                legal_actions=[Action.MOVE, Action.IDLE],
                preferred_action=preferred_action,
                explore=training,
            )

        reward = 0.0
        actual_action = action_name(chosen_action)
        target_label = env.target_description(preferred_target)
        target_position = preferred_target.position if preferred_target is not None else None
        path_length = 0
        status = ""
        previous_position = self.position
        before_remaining = env.remaining_tasks()

        if chosen_action == Action.MOVE:
            next_position, path_length, target_label, target_position, status = self._move_with_path_planning(env, preferred_target)
            if next_position != self.position:
                reward -= 1.0  # Energy cost per movement step for more realistic optimization.
                previous_distance = self._distance_to_target(previous_position, preferred_target)
                next_distance = self._distance_to_target(next_position, preferred_target)
                reward += env.movement_reward(previous_distance, next_distance)
                self.movement_actions += 1
                if next_distance >= previous_distance:
                    self.useless_moves += 1
                if preferred_target is not None:
                    straight_line = manhattan_distance(previous_position, preferred_target.position)
                    if path_length > straight_line:
                        self.avoided_collisions += 1
                if env.observe(next_position).nearby_obstacles > 0:
                    self.avoided_collisions += 1
                self.heading = (next_position[0] - previous_position[0], next_position[1] - previous_position[1])
            elif status in ("planned route blocked", "no valid move"):
                reward += env.collision_penalty()
            self.position = next_position
            actual_action = "MOVE"
        elif chosen_action in (Action.WATER, Action.REMOVE_WEED, Action.IDLE):
            reward_delta, status, completed, task_kind = env.execute_task(self.position, chosen_action)
            reward += reward_delta
            if completed and task_kind == "Weed":
                self.weeds_removed += 1
                self.tasks_completed += 1
            if completed and task_kind == "Dry Soil":
                self.cells_watered += 1
                self.tasks_completed += 1
            if chosen_action == Action.IDLE:
                self.idle_actions += 1
            actual_action = action_name(chosen_action)
        else:
            reward += -1.0

        env.update_dynamics()
        self.step_count += 1
        self.total_steps += 1
        done = env.remaining_tasks() == 0

        learning_reward = reward

        next_state = env.observe(self.position).state
        self.last_action = chosen_action
        self.recent_positions.append(self.position)

        result = RoverStepResult(
            reward=reward,
            learning_reward=learning_reward,
            action=chosen_action,
            actual_action=actual_action,
            target=target_label,
            target_position=target_position,
            path_length=path_length,
            status=status,
            done=done,
        )
        return state, next_state, result

    def _distance_to_target(self, position: Cell, target: Optional[TargetInfo]) -> int:
        if target is None:
            return 0
        return manhattan_distance(position, target.position)

    def _move_with_path_planning(self, env: FarmEnvironment, target: Optional[TargetInfo]) -> tuple[Cell, int, str, Optional[Cell], str]:
        if target is None:
            return self._explore_one_step(env)

        path = target.path
        if len(path) < 2:
            return self._explore_one_step(env)

        self.current_target = target.position
        self.current_path = path
        next_position = path[1]
        if manhattan_distance(self.position, next_position) != 1:
            return self._explore_one_step(env)

        if CellType(env.grid[next_position]) == CellType.OBSTACLE:
            self.collisions += 1
            return self.position, 0, "Obstacle", self.current_target, "planned route blocked"

        return next_position, len(path) - 1, env.target_description(target), self.current_target, "path planned"

    def _explore_one_step(self, env: FarmEnvironment) -> tuple[Cell, int, str, Optional[Cell], str]:
        x, y = self.position
        candidate_positions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        valid = [pos for pos in candidate_positions if env.walkable(pos)]
        if not valid:
            self.collisions += 1
            return self.position, 0, "None", None, "no valid move"

        distances = []
        for pos in valid:
            scanned = env.scan_targets(pos)[2]
            path_length = scanned.distance if scanned is not None else 10**6
            revisit_penalty = 2 if pos in self.recent_positions else 0
            distances.append((path_length + revisit_penalty, pos, scanned))
        distances.sort(key=lambda item: item[0])
        next_position = distances[0][1]
        chosen_target = distances[0][2]
        if chosen_target is not None:
            self.current_target = chosen_target.position
            self.current_path = chosen_target.path
        return next_position, 1, env.target_description(chosen_target), self.current_target if chosen_target else None, "exploring"

    def step_label(self, max_steps: int) -> str:
        return f"{self.step_count}/{max_steps}"

    def _target_label(self, env: FarmEnvironment, target: Optional[Cell]) -> str:
        if target is None:
            return "None"
        cell_type = CellType(env.grid[target])
        if cell_type == CellType.WEED:
            return "Weed"
        if cell_type == CellType.DRY_SOIL:
            return "Dry Soil"
        return "Target"
