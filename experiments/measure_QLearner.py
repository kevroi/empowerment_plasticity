import numpy as np
from src.environments.FourRooms import FourRooms
from src.agents.QLearning import QLearning

# Hyperparameters
NUM_EPISODES = 100
MAX_STEPS = 500
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.1

# Env and Agent
env = FourRooms(max_steps=MAX_STEPS)
agent = QLearning(
    n_actions=env.action_space.n,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON
)

# Logging
returns_per_episode = []
obs_seqs_per_episode = []
action_seqs_per_episode = []

for episode in range(NUM_EPISODES):
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
