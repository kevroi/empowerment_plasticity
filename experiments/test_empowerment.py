import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import List, Tuple, Dict
import sys
sys.path.append('.')  # Add current directory to path
from src.environment import LightRooms
import src.mutual_info as mi

def collect_action_observation_sequences(env, room_id: int, n_episodes: int = 1000, 
                                       episode_length: int = 50) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Collect action and observation sequences for a specific room using strategic action patterns.
    
    The key insight is that empowerment measures how much the agent's actions influence
    future observations. We need action sequences that:
    1. Vary systematically to test causal influence
    2. Stay in the target room to measure room-specific empowerment
    3. Provide enough variation to estimate directed information
    
    Args:
        env: The LightRooms environment
        room_id: Which room to stay in
        n_episodes: Number of episodes to collect
        episode_length: Length of each episode
    
    Returns:
        Tuple of (action_sequences, observation_sequences)
    """
    action_sequences = []
    observation_sequences = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        current_room, light_state = obs
        
        # Move to target room efficiently
        while current_room != room_id:
            if current_room < room_id:
                action = 1  # Move right
            else:
                action = 0  # Move left
            obs, _, _, _, _ = env.step(action)
            current_room, light_state = obs
        
        # Now collect sequences in the target room using strategic patterns
        episode_actions = []
        episode_observations = []
        
        # Strategy: Use different action patterns to maximize information
        # Pattern 1: Frequent lever pulls (high intervention)
        # Pattern 2: Sparse lever pulls (low intervention) 
        # Pattern 3: Systematic patterns (to test specific causal effects)
        
        pattern_type = episode % 4
        
        for step in range(episode_length):
            # if pattern_type == 0:
            #     # High intervention pattern - pull lever frequently
            #     action = 2 if np.random.random() < 0.8 else np.random.choice([0, 1, 2])
            # elif pattern_type == 1:
            #     # Low intervention pattern - pull lever rarely
            #     action = 2 if np.random.random() < 0.2 else np.random.choice([0, 1, 2])
            # elif pattern_type == 2:
            #     # Systematic pattern - alternating high/low intervention
            #     action = 2 if (step // 5) % 2 == 0 else np.random.choice([0, 1, 2])
            # else:
            #     # Random pattern for baseline
            #     action = np.random.choice([0, 1, 2])
            
            # # Ensure we stay in target room
            # if action == 0 and room_id == 0:
            #     action = 2  # Can't go left from room 0
            # elif action == 1 and room_id == env.n_rooms:
            #     action = 2  # Can't go right from last room
            # elif action in [0, 1]:
            #     # For interior rooms, sometimes stay (pull lever) to avoid leaving
            #     if np.random.random() < 0.3:  # 30% chance to stay instead of move
            #         action = 2

            action = np.random.choice([0, 1, 2])
            
            episode_actions.append(action)
            
            obs, _, _, _, _ = env.step(action)
            current_room, light_state = obs
            
            # Record the light state as observation
            episode_observations.append(light_state)
            
            # If we moved rooms, immediately return to target room (don't record this step)
            if current_room != room_id:
                while current_room != room_id:
                    if current_room < room_id:
                        obs, _, _, _, _ = env.step(1)  # Move right
                    else:
                        obs, _, _, _, _ = env.step(0)  # Move left
                    current_room, light_state = obs
        
        action_sequences.append(episode_actions)
        observation_sequences.append(episode_observations)
    
    return action_sequences, observation_sequences

def calculate_windowed_directed_info(action_seqs: List[List[int]], 
                                   obs_seqs: List[List[int]], 
                                   window_size: int = 5) -> float:
    """
    Calculate directed information using windowed approach to avoid sparse estimation.
    """
    if len(action_seqs) == 0 or len(obs_seqs) == 0:
        return 0.0
    
    total_di = 0.0
    count = 0
    
    sequence_length = len(action_seqs[0])
    
    for t in range(window_size, sequence_length):
        # Extract windowed histories
        action_histories = []
        obs_nows = []
        obs_histories = []
        
        for i in range(len(action_seqs)):
            # Action history: A_{t-window_size+1} to A_t
            action_hist = tuple(action_seqs[i][t-window_size+1:t+1])
            action_histories.append(action_hist)
            
            # Current observation: O_t
            obs_nows.append(obs_seqs[i][t])
            
            # Observation history: O_{t-window_size} to O_{t-1}
            obs_hist = tuple(obs_seqs[i][t-window_size:t])
            obs_histories.append(obs_hist)
        
        # Calculate conditional mutual information: I(A^t; O_t | O^{t-1})
        cmi = mi.conditional_mutual_info(action_histories, obs_nows, obs_histories)
        total_di += cmi
        count += 1
    
    return total_di / count if count > 0 else 0.0

def measure_empowerment_per_room(n_rooms: int = 4,
                                 n_episodes: int = 1000, 
                                 episode_length: int = 30):
    """
    Measure agent's empowerment(directed information from actions to observations)
    and plasticity (directed information from observations to actions) for each room.
    
    Args:
        n_rooms: Number of rooms (excluding room 0)
        n_episodes: Number of episodes per room
        episode_length: Length of each episode
    
    Returns:
        Dictionary mapping room_id to empowerment and plasticity values
    """
    env = LightRooms(n_rooms=n_rooms)
    
    empowerment_values = {}
    plasticity_values = {}
    control_probabilities = {}
    
    for room_id in range(n_rooms + 1):  # 0 to n_rooms
        print(f"Collecting data for room {room_id}...")
        
        # Collect action-observation sequences
        action_seqs, obs_seqs = collect_action_observation_sequences(
            env, room_id, n_episodes, episode_length
        )
        
        # empowerment = mi.directed_info(action_seqs, obs_seqs) / episode_length
        # plasticity = mi.directed_info(obs_seqs, action_seqs) / episode_length

        empowerment = calculate_windowed_directed_info(action_seqs, obs_seqs)
        plasticity = calculate_windowed_directed_info(obs_seqs, action_seqs)
        
        empowerment_values[room_id] = empowerment
        plasticity_values[room_id] = plasticity
        control_probabilities[room_id] = room_id / n_rooms
        
        print(f"Room {room_id}: Empowerment = {empowerment:.4f}, Plasticity = {plasticity:.4f}, Control Prob = {control_probabilities[room_id]:.2f}")
    
    env.close()
    return empowerment_values, plasticity_values, control_probabilities


def plot_empowerment_vs_room(empowerment_values: Dict[int, float],
                             plasticity_values: Dict[int, float],
                             control_probabilities: Dict[int, float]):
    """
    Plot empowerment as a function of room number.
    """
    rooms = sorted(empowerment_values.keys())
    empowerment_vals = [empowerment_values[room] for room in rooms]
    plasticity_vals = [plasticity_values[room] for room in rooms]
    control_probs = [control_probabilities[room] for room in rooms]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    
    # Plot empowerment
    ax1.plot(rooms, empowerment_vals, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Room Number')
    ax1.set_ylabel('Empowerment')
    ax1.set_title('Empowerment vs Room Number')
    ax1.grid(True, alpha=0.3)

    ax2.plot(rooms, plasticity_vals, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Room Number')
    ax2.set_ylabel('Plasticity')
    ax2.set_title('Plasticity vs Room Number')
    ax2.grid(True, alpha=0.3)
    
    # Plot control probability for reference
    ax3.plot(rooms, control_probs, 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('Room Number')
    ax3.set_ylabel('Control Probability')
    ax3.set_title('Control Probability vs Room Number')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def run_empowerment_experiment():
    """
    Main function to run the empowerment measurement experiment.
    """
    print("Starting empowerment measurement experiment...")
    
    # Run experiment
    empowerment_values, plasticity_values, control_probabilities = measure_empowerment_per_room(
        n_rooms=19,
        n_episodes=1000,
        episode_length=30
    )
    
    # Plot results
    fig = plot_empowerment_vs_room(empowerment_values, plasticity_values, control_probabilities)
    
    # Print summary
    print("\n=== RESULTS ===")
    for room in sorted(empowerment_values.keys()):
        print(f"Room {room}: Empowerment = {empowerment_values[room]:.4f} bits, "
              f"Control Prob = {control_probabilities[room]:.2f}")
    
    return empowerment_values, plasticity_values, control_probabilities, fig

if __name__ == "__main__":
    run_empowerment_experiment()
    
    # print("Please import your LightRooms environment and information theory module first!")