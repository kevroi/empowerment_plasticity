# environments/four_rooms_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FourRooms(gym.Env):
    metadata = {"render_modes": ["human"],
                "render_fps": 4}

    def __init__(self,
                 render_mode=None,
                 max_steps=500,
                 goal_index=0):
        super().__init__()
        self.grid_size = 11
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.goal_positions = [(9, 9), (1, 1), (1, 9), (9, 1)]
        self.goal_index = goal_index

        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.action_space = spaces.Discrete(4)  # up, right, down, left

        self.agent_pos = None
        self.goal_pos = self.goal_positions[self.goal_index]
        self.steps_taken = 0

        self._build_walls()
        self.reset()

    def _build_walls(self):
        self.walls = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        # Add internal walls to create four rooms
        self.walls[5, 1:5] = True
        self.walls[5, 6:] = True
        self.walls[1:5, 5] = True
        self.walls[6:, 5] = True
        # Add doors
        self.walls[5, 3] = False
        self.walls[5, 7] = False
        self.walls[3, 5] = False
        self.walls[7, 5] = False

    def _pos_to_state(self, pos):
        return pos[0] * self.grid_size + pos[1]

    def _state_to_pos(self, state):
        return (state // self.grid_size, state % self.grid_size)

    def reset(self, seed=None, options=None, goal_index=0):
        super().reset(seed=seed)
        self.agent_pos = (0, 0)
        self.steps_taken = 0
        self.goal_index = goal_index
        self.goal_pos = self.goal_positions[self.goal_index]
        return self._pos_to_state(self.agent_pos), {}

    def step(self, action):
        self.steps_taken += 1
        move = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        next_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        if (
            0 <= next_pos[0] < self.grid_size
            and 0 <= next_pos[1] < self.grid_size
            and not self.walls[next_pos]
        ):
            self.agent_pos = next_pos

        done = self.agent_pos == self.goal_pos
        truncated = self.steps_taken >= self.max_steps
        reward = 1.0 if done else 0.00

        return self._pos_to_state(self.agent_pos), reward, done, truncated, {}

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        grid[self.walls] = "#"
        # Mark all possible goal positions with 'o'
        for pos in self.goal_positions:
            grid[pos] = "o"
        # Mark the current goal with 'G' (overwrites 'o' if overlapping)
        grid[self.goal_pos] = "G"
        # Mark the agent with 'A' (overwrites any symbol if overlapping)
        grid[self.agent_pos] = "A"
        print("\n".join("".join(row) for row in grid))
        print()
