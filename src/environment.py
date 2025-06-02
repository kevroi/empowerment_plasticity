import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

class LightRooms(gym.Env):
    """
    An environment implementing the empowerment/plasticity rooms example.
    
    The environment consists of n+1 rooms, each with a switch and a light.
    The agent can move between rooms or pull the lever in the current room.
    Each room has a different probability of controlling the light state.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": 4}
    
    def __init__(self, n_rooms: int = 4, render_mode: Optional[str] = None):
        """
        Initialize the environment.
        
        Args:
            n_rooms: Number of rooms (excluding room 0). Total rooms will be n_rooms + 1
            render_mode: The render mode to use
        """
        super().__init__()
        
        self.n_rooms = n_rooms
        self.n_total_rooms = n_rooms + 1
        self.render_mode = render_mode
        
        # Action space: 0 = move left, 1 = move right, 2 = pull lever
        self.action_space = spaces.Discrete(3)
        
        # Observation space: (room_index, light_state)
        # room_index: 0 to n_rooms
        # light_state: 0 (off) or 1 (on)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.n_total_rooms),
            spaces.Discrete(2)
        ))
        
        # Environment state
        self.current_room = 0
        self.light_state = 0
        self.x_state = 0  # Bernoulli state that determines light behavior
        self.pending_switch = None  # can be 'flip', 'random', or None - may need a better typing system

        # For visualization
        self.window = None
        self.clock = None
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Tuple[int, int], Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_room = 0
        self.light_state = self.np_random.integers(0, 2)  # Random initial light state
        self.x_state = self.np_random.integers(0, 2)  # Random initial X state
        
        observation = (self.current_room, self.light_state)
        info = {"x_state": self.x_state}
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: 0 = move left, 1 = move right, 2 = pull lever
            
        Returns:
            observation: (room_index, light_state)
            reward: Always 0 for now
            terminated: Always False
            truncated: Always False
            info: Additional information
        """
        # Apply pending effect from previous lever pull
        if self.pending_switch == "flip":
            self.light_state = 1 - self.light_state
        elif self.pending_switch == "random":
            self.light_state = self.x_state
        self.pending_switch = None  # reset

        # Update X state
        self.x_state = self.np_random.integers(0, 2)

        # Movement actions
        if action == 0:
            self.current_room = max(0, self.current_room - 1)
        elif action == 1:
            self.current_room = min(self.n_total_rooms - 1, self.current_room + 1)
        elif action == 2:
            control_prob = self.current_room / self.n_rooms
            if self.np_random.random() < control_prob:
                self.pending_switch = "flip"
            else:
                self.pending_switch = "random"

        observation = (self.current_room, self.light_state)
        info = {"x_state": self.x_state, "control_probability": self.current_room / self.n_rooms}
        return observation, 0.0, False, False, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        """Render a single frame of the environment."""
        import pygame
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((800, 400))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        canvas = pygame.Surface((800, 400))
        canvas.fill((255, 255, 255))
        
        # Draw rooms
        room_width = 700 / self.n_total_rooms
        for i in range(self.n_total_rooms):
            x = 50 + i * room_width
            y = 100
            width = room_width - 10
            height = 200
            
            # Draw room
            color = (200, 200, 200) if i != self.current_room else (150, 150, 255)
            pygame.draw.rect(canvas, color, (x, y, width, height))
            
            # Draw light
            light_color = (255, 255, 0) if (i == self.current_room and self.light_state == 1) else (100, 100, 100)
            pygame.draw.circle(canvas, light_color, (int(x + width/2), int(y + 50)), 20)
            
            # Draw room number
            font = pygame.font.Font(None, 36)
            text = font.render(str(i), True, (0, 0, 0))
            canvas.blit(text, (x + width/2 - 10, y + height - 40))
            
            # Draw control probability
            prob = i / self.n_rooms
            prob_text = font.render(f"{prob:.2f}", True, (0, 0, 0))
            canvas.blit(prob_text, (x + width/2 - 20, y + 10))
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
    
    def close(self):
        """Clean up resources."""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None 