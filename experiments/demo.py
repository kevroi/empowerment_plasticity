import gymnasium as gym
import numpy as np
from src.environments.LightRooms import LightRooms

def main():
    N = 4 # number of rooms = N+1
    NUM_STEPS = 100

    env = LightRooms(n_rooms=N, render_mode="human")
    
    # Reset the environment
    observation, info = env.reset()
    print(f"Initial observation: Room {observation[0]}, Light {'on' if observation[1] else 'off'}")
    print(f"Initial X state: {info['x_state']}")
    
    # Run for 100 steps
    for step in range(NUM_STEPS):
        # Uniform Random Policy
        action = env.action_space.sample()
        action_names = ["move left", "move right", "pull lever"]
        
        # Take step
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Print information
        print(f"\nStep {step + 1}:")
        print(f"Action: {action_names[action]}")
        print(f"New observation: Room {observation[0]}, Light {'on' if observation[1] else 'off'}")
        print(f"X state: {info['x_state']}")
        print(f"Control probability in current room: {info['control_probability']:.2f}")
        
        if terminated or truncated:
            break
    
    env.close()

if __name__ == "__main__":
    main()
