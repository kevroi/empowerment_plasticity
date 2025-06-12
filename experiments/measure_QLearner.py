import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.environments.FourRooms import FourRooms
from src.agents.QLearning import QLearning


def run(env, agent, num_episodes):
    # Logging
    returns_per_episode = []
    obs_seqs_per_episode = []
    action_seqs_per_episode = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = truncated = False
        total_reward = 0

        obs_seq = []
        action_seq = []

        while not (done or truncated):
            action = agent.act(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            agent.learn(obs, action, reward, next_obs, done, truncated)

            total_reward += reward
            obs_seq.append(obs)
            action_seq.append(action)
            obs = next_obs

        # Logging episode data
        returns_per_episode.append(total_reward)
        obs_seqs_per_episode.append(obs_seq)
        action_seqs_per_episode.append(action_seq)

        # print(f"Episode {episode + 1}: Return = {total_reward:.2f}")

    # print("Training complete!")
    return returns_per_episode, obs_seqs_per_episode, action_seqs_per_episode

# Moving average to smooth the curve (optional)
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


if __name__ == "__main__":
    # Hyperparameters
    NUM_EPISODES = 100
    MAX_STEPS = 500
    NUM_RUNS = 50
    ALPHA = 0.1
    GAMMA = 0.99
    EPSILON = 0.1
    all_returns = []

    
    for _ in tqdm(range(NUM_RUNS)):
        env = FourRooms(max_steps=MAX_STEPS)
        agent = QLearning(
            n_actions=env.action_space.n,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon=EPSILON
        )
        returns, _, _ = run(env, agent, NUM_EPISODES)
        all_returns.append(returns)

    # Convert to array: shape = (NUM_RUNS, NUM_EPISODES)
    all_returns = np.array(all_returns)
    mean_returns = np.mean(all_returns, axis=0)
    # smoothed_mean_returns = moving_average(mean_returns, window_size=10)
    stderr = np.std(all_returns, axis=0) / np.sqrt(NUM_RUNS)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(mean_returns, label=f"Avg return over {NUM_RUNS} runs (smoothed)", linewidth=2)
    plt.fill_between(
        range(len(mean_returns)),
        mean_returns - stderr,
        mean_returns + stderr,
        alpha=0.2
    )
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"Q-Learning in FourRooms (Average of {NUM_RUNS} Runs)")
    plt.legend()
    plt.tight_layout()
    plt.show()