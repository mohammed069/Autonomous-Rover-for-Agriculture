"""Simple Q-learning agent used by the rover."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import DefaultDict, Iterable, Sequence

import numpy as np

from utils import Action


class QLearningAgent:
    def __init__(
        self,
        action_space: Sequence[Action] | None = None,
        learning_rate: float = 0.2,
        discount_factor: float = 0.92,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.985,
    ) -> None:
        self.action_space = list(action_space or list(Action))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: DefaultDict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(len(self.action_space), dtype=float))

    def select_action(
        self,
        state: tuple,
        legal_actions: Iterable[Action] | None = None,
        preferred_action: Action | None = None,
        explore: bool = True,
    ) -> Action:
        legal = list(legal_actions or self.action_space)
        if not legal:
            legal = list(self.action_space)

        if explore and random.random() < self.epsilon:
            if preferred_action in legal and random.random() < 0.7:
                return preferred_action  # type: ignore[return-value]
            return random.choice(legal)

        q_values = self.q_table[state].copy()
        masked = np.full(len(self.action_space), -1e9, dtype=float)
        for action in legal:
            action_index = int(action)
            masked[action_index] = q_values[action_index]
            if preferred_action is not None:
                masked[action_index] += 0.4
                if action == preferred_action:
                    masked[action_index] += 1.0

        best_index = int(np.argmax(masked))
        return self.action_space[best_index]

    def update(self, state: tuple, action: Action, reward: float, next_state: tuple, done: bool) -> None:
        current_q = self.q_table[state][int(action)]
        next_best = 0.0 if done else float(np.max(self.q_table[next_state]))
        target = reward + self.discount_factor * next_best
        self.q_table[state][int(action)] = current_q + self.learning_rate * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def set_greedy(self) -> None:
        self.epsilon = 0.0
