"""Entry point for the Autonomous Rover for Agriculture project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from config import (
    DISCOUNT_FACTOR,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    GRID_SIZE,
    LEARNING_RATE,
    MAX_STEPS,
    MAX_STEPS_PER_EPISODE,
    TRAINING_VISUALIZATION,
    TRAINING_EPISODES,
)
from environment import FarmEnvironment
from qlearning import QLearningAgent
from rover import Rover
from utils import Action, seed_everything
from visualization import FarmVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Rover for Agriculture")
    parser.add_argument("--episodes", type=int, default=TRAINING_EPISODES, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=MAX_STEPS_PER_EPISODE, help="Maximum steps per episode")
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE, help="Farm grid size")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--skip-simulation", action="store_true", help="Train only and disable live visualization")
    return parser.parse_args()


def _clamped_efficiency(tasks_completed: int, total_task_budget: int) -> float:
    if total_task_budget <= 0:
        return 0.0
    calculated = (tasks_completed / total_task_budget) * 100.0
    return max(0.0, min(calculated, 100.0))


def train(agent: QLearningAgent, env: FarmEnvironment, rover: Rover, episodes: int, max_steps: int, visualizer: FarmVisualizer | None = None) -> tuple[List[float], int, Dict[str, float]]:
    reward_history: List[float] = []
    task_budget = 0
    total_reward = 0.0
    success_count = 0
    tasks_per_episode: List[int] = []
    collision_delta_per_episode: List[int] = []

    for episode in range(1, episodes + 1):
        start_position = env.reset()
        task_budget += env.initial_task_count
        rover.reset(start_position)
        episode_reward = 0.0
        episode_action = Action.IDLE
        episode_target = "None"
        episode_path_length = 0
        episode_step_reward = 0.0
        start_tasks_completed = rover.tasks_completed
        start_collisions = rover.collisions
        episode_success = False

        for step_index in range(max_steps):
            state, next_state, result = rover.step(env, agent, training=True)
            agent.update(state, result.action, result.learning_reward, next_state, result.done)
            episode_reward += result.reward
            episode_action = result.action
            episode_target = result.target
            episode_path_length = result.path_length
            episode_step_reward = result.reward
            if visualizer is not None:
                if not visualizer.handle_events():
                    visualizer.close()
                    raise KeyboardInterrupt
                visualizer.draw(
                    env=env,
                    rover_position=rover.position,
                    rover_heading=rover.heading,
                    episode=episode,
                    current_step=f"{step_index + 1}/{max_steps}",
                    episode_reward=episode_reward,
                    total_reward=total_reward + episode_reward,
                    current_action=result.actual_action,
                    tasks_completed=rover.tasks_completed,
                    current_target=result.target,
                    efficiency=_clamped_efficiency(rover.tasks_completed, task_budget),
                    collision_rate=(rover.collisions / max(1, rover.total_steps)) * 100.0,
                    target_position=result.target_position,
                    path=rover.current_path,
                    reward_history=reward_history if reward_history else [0.0],
                    learning_progress=episode / max(1, episodes),
                    average_reward=(sum(reward_history) / max(1, len(reward_history))) if reward_history else 0.0,
                )
            if result.done:
                episode_success = True
                break

        agent.decay_epsilon()
        total_reward += episode_reward
        rover.total_reward += episode_reward
        if episode_success:
            success_count += 1

        tasks_per_episode.append(rover.tasks_completed - start_tasks_completed)
        collision_delta_per_episode.append(rover.collisions - start_collisions)
        reward_history.append(episode_reward)
        print(f"Episode {episode:03d}")
        print(f"Action: {episode_action.name}")
        print(f"Reward: {episode_step_reward:+.1f}")
        print(f"Target: {episode_target}")
        print(f"Path Length: {episode_path_length}")
        print(f"Tasks Remaining: {env.remaining_tasks()}")
        print(f"Current Step: {rover.step_count}/{max_steps}")
        print(f"Episode Reward: {episode_reward:+.1f}")
        print(f"Total Reward: {total_reward:+.1f}")
        print()

    last_ten = reward_history[-10:] if reward_history else [0.0]
    average_last_ten = sum(last_ten) / len(last_ten)
    success_rate = (success_count / max(1, episodes)) * 100.0
    avg_tasks = sum(tasks_per_episode) / max(1, len(tasks_per_episode))
    collision_rate = (sum(collision_delta_per_episode) / max(1, len(collision_delta_per_episode)))
    learning_stats = {
        "avg_reward_last_10": average_last_ten,
        "success_rate": success_rate,
        "tasks_per_episode": avg_tasks,
        "collision_rate": collision_rate,
    }

    return reward_history, task_budget, learning_stats


def plot_training_results(reward_history: List[float]) -> Path:
    output_path = Path("training_results.png")
    episodes = list(range(1, len(reward_history) + 1))
    rolling_average = []
    window = 10
    for index in range(len(reward_history)):
        start = max(0, index - window + 1)
        rolling_average.append(sum(reward_history[start : index + 1]) / (index - start + 1))

    plt.figure(figsize=(11, 5.5))
    plt.plot(episodes, reward_history, color="#3d8b40", linewidth=1.3, alpha=0.75, label="Episode reward")
    plt.plot(episodes, rolling_average, color="#f1c453", linewidth=2.7, label="Moving average (10 episodes)")
    plt.title("Autonomous Rover Training Progress", pad=12)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.2, linestyle="--", linewidth=0.8)
    plt.margins(x=0.01)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def print_summary(reward_history: List[float], rover: Rover, total_task_budget: int, learning_stats: Dict[str, float]) -> None:
    average_reward = sum(reward_history) / max(1, len(reward_history))
    efficiency = _clamped_efficiency(rover.tasks_completed, total_task_budget)
    collision_rate = (rover.collisions / max(1, rover.total_steps)) * 100.0

    print("Final Simulation Summary")
    print(f"Total Tasks Completed: {rover.tasks_completed}")
    print(f"Final Reward: {rover.total_reward:+.1f}")
    print(f"Average Reward: {average_reward:+.1f}")
    print(f"Efficiency: {efficiency:.1f}%")
    print(f"Total Weeds Removed: {rover.weeds_removed}")
    print(f"Total Cells Watered: {rover.cells_watered}")
    print(f"Collisions Handled: {rover.collisions}")
    print(f"Avoided Collisions: {rover.avoided_collisions}")
    print(f"Collision Rate: {collision_rate:.1f}%")
    print(f"Avg Reward (Last 10 Episodes): {learning_stats['avg_reward_last_10']:+.1f}")
    print(f"Success Rate: {learning_stats['success_rate']:.1f}%")
    print(f"Avg Tasks per Episode: {learning_stats['tasks_per_episode']:.2f}")
    print(f"Avg Collisions per Episode: {learning_stats['collision_rate']:.2f}")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    env = FarmEnvironment(grid_size=args.grid_size, seed=args.seed)
    rover = Rover(position=(0, 0))
    rover.reset(env.reset())

    agent = QLearningAgent(
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
    )

    visualizer = FarmVisualizer(env.grid_size) if (TRAINING_VISUALIZATION and not args.skip_simulation) else None
    try:
        reward_history, training_task_budget, learning_stats = train(
            agent,
            env,
            rover,
            episodes=args.episodes,
            max_steps=args.steps,
            visualizer=visualizer,
        )
    finally:
        if visualizer is not None:
            visualizer.close()

    graph_path = plot_training_results(reward_history)
    print(f"Training graph saved to: {graph_path.resolve()}")

    print_summary(reward_history, rover, training_task_budget, learning_stats)


if __name__ == "__main__":
    main()
