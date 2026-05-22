"""Pygame-based visualization for the farm rover simulation."""

from __future__ import annotations

from typing import Sequence

import pygame

from config import BACKGROUND_COLOR, CELL_SIZE, COLOR_MAP, GRID_LINE_COLOR, PANEL_WIDTH, SIMULATION_FPS_DEFAULT, SIMULATION_FPS_MAX, SIMULATION_FPS_MIN, TEXT_COLOR
from environment import FarmEnvironment
from utils import Cell, CellType, format_position


class FarmVisualizer:
    def __init__(self, grid_size: int, cell_size: int = CELL_SIZE, panel_width: int = PANEL_WIDTH) -> None:
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.panel_width = panel_width
        self.width = grid_size * cell_size + panel_width
        self.height = grid_size * cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Autonomous Rover for Agriculture")
        self.font = pygame.font.SysFont("arial", 18)
        self.large_font = pygame.font.SysFont("arial", 24, bold=True)
        self.small_font = pygame.font.SysFont("arial", 14)
        self.clock = pygame.time.Clock()
        self.simulation_fps = SIMULATION_FPS_DEFAULT

    def draw(
        self,
        env: FarmEnvironment,
        rover_position: Cell,
        rover_heading: Cell,
        episode: int,
        current_step: str,
        episode_reward: float,
        total_reward: float,
        current_action: str,
        tasks_completed: int,
        current_target: str,
        efficiency: float,
        collision_rate: float,
        target_position: Cell | None,
        path: Sequence[Cell],
        reward_history: Sequence[float],
        learning_progress: float,
        average_reward: float,
    ) -> None:
        self.clock.tick(self.simulation_fps)
        self.screen.fill(BACKGROUND_COLOR)
        self._draw_grid(env, path, target_position)
        self._draw_rover(rover_position, rover_heading)
        self._draw_panel(env, episode, current_step, episode_reward, total_reward, current_action, tasks_completed, current_target, rover_position, efficiency, collision_rate, reward_history, learning_progress, average_reward)
        pygame.display.flip()

    def _draw_grid(self, env: FarmEnvironment, path: Sequence[Cell], target_position: Cell | None) -> None:
        path_set = set(path)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                cell_type = CellType(env.grid[x, y])
                color = COLOR_MAP[cell_type.name]
                pygame.draw.rect(self.screen, color, rect)
                if (x, y) in path_set:
                    pygame.draw.rect(self.screen, COLOR_MAP["PATH"], rect.inflate(-18, -18), width=0, border_radius=6)
                if target_position == (x, y):
                    pygame.draw.rect(self.screen, (255, 94, 94), rect.inflate(-5, -5), width=3, border_radius=6)
                pygame.draw.rect(self.screen, GRID_LINE_COLOR, rect, width=1)
                self._draw_symbol(cell_type, rect)

        pygame.draw.rect(self.screen, (110, 120, 110), pygame.Rect(0, 0, self.grid_size * self.cell_size, self.grid_size * self.cell_size), width=2)

    def _draw_symbol(self, cell_type: CellType, rect: pygame.Rect) -> None:
        symbol_map = {
            CellType.CROP: "C",
            CellType.WEED: "W",
            CellType.DRY_SOIL: "D",
            CellType.OBSTACLE: "#",
            CellType.WATERED_SOIL: "~",
            CellType.EMPTY: " ",
        }
        symbol = symbol_map[cell_type]
        if symbol.strip():
            label = self.font.render(symbol, True, (240, 240, 240))
            self.screen.blit(label, label.get_rect(center=rect.center))

    def _draw_rover(self, rover_position: Cell, rover_heading: Cell) -> None:
        x, y = rover_position
        rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.ellipse(self.screen, COLOR_MAP["ROVER"], rect.inflate(-10, -10))
        label = self.large_font.render("R", True, (24, 24, 24))
        self.screen.blit(label, label.get_rect(center=rect.center))

        cx, cy = rect.center
        dx, dy = rover_heading[1], rover_heading[0]
        end_pos = (cx + dx * 12, cy + dy * 12)
        pygame.draw.line(self.screen, (30, 30, 30), (cx, cy), end_pos, width=3)
        pygame.draw.circle(self.screen, (30, 30, 30), end_pos, 4)

    def _draw_panel(
        self,
        env: FarmEnvironment,
        episode: int,
        current_step: str,
        episode_reward: float,
        total_reward: float,
        current_action: str,
        tasks_completed: int,
        current_target: str,
        rover_position: Cell,
        efficiency: float,
        collision_rate: float,
        reward_history: Sequence[float],
        learning_progress: float,
        average_reward: float,
    ) -> None:
        x_offset = self.grid_size * self.cell_size + 18
        panel = pygame.Rect(self.grid_size * self.cell_size, 0, self.panel_width, self.height)
        pygame.draw.rect(self.screen, (23, 31, 25), panel)
        pygame.draw.line(self.screen, (70, 95, 76), (self.grid_size * self.cell_size, 0), (self.grid_size * self.cell_size, self.height), width=2)

        lines = [
            ("Episode", str(episode)),
            ("Current Step", current_step),
            ("Weather", env.weather),
            ("Episode Reward", f"{episode_reward:+.1f}"),
            ("Total Reward", f"{total_reward:+.1f}"),
            ("Action", current_action),
            ("Target", current_target),
            ("Rover Position", format_position(rover_position)),
            ("Average Reward", f"{average_reward:.1f}"),
            ("Efficiency", f"{efficiency:.1f}%"),
            ("Collision Rate", f"{collision_rate:.1f}%"),
            ("Tasks Left", str(env.remaining_tasks())),
            ("Tasks Done", str(tasks_completed)),
            ("Learning Progress", f"{learning_progress * 100:.0f}%"),
        ]
        y = 18
        title = self.large_font.render("Farm Rover AI", True, TEXT_COLOR)
        self.screen.blit(title, (x_offset, y))
        y += 40

        for label, value in lines:
            text = self.font.render(f"{label}: {value}", True, TEXT_COLOR)
            self.screen.blit(text, (x_offset, y))
            y += 30

        self._draw_progress_bar(x_offset, y + 8, self.panel_width - 36, learning_progress)
        self._draw_legend(x_offset, y + 50)
        self._draw_reward_sparkline(reward_history, x_offset, y + 116)

    def _draw_progress_bar(self, x_offset: int, y_offset: int, width: int, progress: float) -> None:
        label = self.small_font.render("Training Progress", True, TEXT_COLOR)
        self.screen.blit(label, (x_offset, y_offset))
        y_offset += 20
        bar_rect = pygame.Rect(x_offset, y_offset, width, 16)
        pygame.draw.rect(self.screen, (50, 60, 54), bar_rect, border_radius=8)
        filled_width = max(0, min(width, int(width * max(0.0, min(1.0, progress)))))
        if filled_width > 0:
            pygame.draw.rect(self.screen, (248, 210, 77), pygame.Rect(x_offset, y_offset, filled_width, 16), border_radius=8)
        pygame.draw.rect(self.screen, (95, 110, 100), bar_rect, width=1, border_radius=8)

    def _draw_legend(self, x_offset: int, y_offset: int) -> None:
        legend_items = [
            ("Crop", COLOR_MAP["CROP"]),
            ("Weed", COLOR_MAP["WEED"]),
            ("Dry Soil", COLOR_MAP["DRY_SOIL"]),
            ("Watered", COLOR_MAP["WATERED_SOIL"]),
        ]
        label = self.small_font.render("Legend", True, TEXT_COLOR)
        self.screen.blit(label, (x_offset, y_offset))
        y = y_offset + 20
        for name, color in legend_items:
            pygame.draw.rect(self.screen, color, pygame.Rect(x_offset, y + 3, 14, 14), border_radius=3)
            text = self.small_font.render(name, True, TEXT_COLOR)
            self.screen.blit(text, (x_offset + 22, y))
            y += 20

    def _draw_reward_sparkline(self, reward_history: Sequence[float], x_offset: int, y_offset: int) -> None:
        title = self.font.render("Training Reward Trend", True, TEXT_COLOR)
        self.screen.blit(title, (x_offset, y_offset))
        y_offset += 26
        width = self.panel_width - 36
        height = 110
        rect = pygame.Rect(x_offset, y_offset, width, height)
        pygame.draw.rect(self.screen, (38, 50, 42), rect, border_radius=8)
        pygame.draw.rect(self.screen, (78, 106, 84), rect, width=1, border_radius=8)

        if len(reward_history) < 2:
            return
        values = list(reward_history[-50:])
        minimum = min(values)
        maximum = max(values)
        span = max(1.0, maximum - minimum)
        points: list[tuple[int, int]] = []
        for index, value in enumerate(values):
            px = x_offset + 10 + int((index / max(1, len(values) - 1)) * (width - 20))
            py = y_offset + height - 10 - int(((value - minimum) / span) * (height - 20))
            points.append((px, py))
        if len(points) > 1:
            pygame.draw.lines(self.screen, (255, 211, 112), False, points, 3)

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.simulation_fps = min(SIMULATION_FPS_MAX, self.simulation_fps + 1)
                elif event.key == pygame.K_DOWN:
                    self.simulation_fps = max(SIMULATION_FPS_MIN, self.simulation_fps - 1)
        return True

    def tick(self, fps: int | None = None) -> None:
        self.clock.tick(fps or self.simulation_fps)

    def close(self) -> None:
        pygame.quit()
