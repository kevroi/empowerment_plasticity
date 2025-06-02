import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional

class BitWorld(gym.Env):
    """
    Environment emits binary observations. Agent receives observation O_t and acts with A_t.
    O_t is a random bit.
    """
    metadata = {"render_modes": []}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)
        self.render_mode = render_mode
        self.rng = np.random.default_rng()
        self.current_obs = self.rng.integers(0, 2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.current_obs = self.rng.integers(0, 2)
        return self.current_obs, {}

    def step(self, action):
        self.current_obs = self.rng.integers(0, 2)
        return self.current_obs, 0.0, False, False, {}
