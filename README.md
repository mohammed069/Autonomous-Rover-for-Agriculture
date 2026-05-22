# Autonomous Rover for Agriculture

An interactive precision-farming simulation where an autonomous rover explores a 2D farm, detects weeds and dry soil, plans routes with A* pathfinding, and improves its strategy over time with Q-learning.

## Project Highlights

- Grid-based farm environment with crops, weeds, dry soil, watered soil, obstacles, and empty land
- Q-learning agent with epsilon-greedy exploration and continuous Q-table updates
- A* search for efficient movement toward weeds and dry soil targets
- Dynamic weather, weed spawning, and soil moisture decay
- Pygame visualization with live rover movement and runtime status panel
- Matplotlib training graph saved as `training_results.png`

## AI Techniques Used

- Reinforcement Learning: Q-learning with reward shaping for watering, weed removal, movement efficiency, and task completion
- Path Planning: A* search to reach the nearest reachable target while avoiding obstacles

## Installation

1. Install Python 3.10+.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## How To Run

Train the rover and then launch the live simulation:

```bash
python main.py
```

Train only and skip the interactive Pygame window:

```bash
python main.py --skip-simulation
```

Optional arguments:

- `--episodes`: number of training episodes
- `--steps`: maximum steps per episode
- `--grid-size`: farm size
- `--seed`: random seed for repeatable runs

## Output Files

- `training_results.png`: reward trend over episodes

## Screenshots

Add simulation screenshots here after running the project locally.

Suggested captures:

- Rover navigating around obstacles
- Weed detection and removal
- Dry soil watering
- Training reward graph

## Future Improvements

- Add persistent model saving and loading for the Q-table
- Extend the rover with multi-target task scheduling
- Add OpenCV-based crop and weed detection overlays
- Experiment with Stable-Baselines3 for deeper reinforcement learning
- Add richer terrain physics and larger farm maps
