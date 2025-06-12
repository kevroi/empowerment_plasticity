import numpy as np
import matplotlib.pyplot as plt
from src.environments.FourRooms import FourRooms
from src.agents.QLearning import QLearning

# Hyperparameters
NUM_EPISODES = 100
MAX_STEPS = 500
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.1


def run(env, agent, num_episodes):
    # Logging
    returns_per_episode = []
    obs_seqs_per_episode = []
    action_seqs_per_episode = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = truncated = False
        total_reward = 0

        obs_seq = [obs]
        action_seq = []

        while not (done or truncated):
            action = agent.act(obs)
            next_obs, reward, done, truncated, _ = env.step(action)

            agent.learn(obs, action, reward, next_obs, done, truncated)

            total_reward += reward
            obs_seq.append(next_obs)
            action_seq.append(action)
            obs = next_obs

        # Logging episode data
        returns_per_episode.append(total_reward)
        obs_seqs_per_episode.append(obs_seq)
        action_seqs_per_episode.append(action_seq)

        print(f"Episode {episode + 1}: Return = {total_reward:.2f}")

    print("Training complete!")
    return returns_per_episode, obs_seqs_per_episode, action_seqs_per_episode

# Moving average to smooth the curve (optional)
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


if __name__ == "__main__":
    # Env and Agent
    env = FourRooms(max_steps=MAX_STEPS)
    agent = QLearning(
        n_actions=env.action_space.n,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON
    )

    # Run agent-env loop
    returns_per_episode, obs_seqs_per_episode, action_seqs_per_episode = run(env, agent, NUM_EPISODES)

    plt.figure(figsize=(10, 5))
    plt.plot(moving_average(returns_per_episode), label="Smoothed (10-episode MA)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Q-Learning in FourRooms")
    plt.legend()
    plt.tight_layout()
    plt.show()