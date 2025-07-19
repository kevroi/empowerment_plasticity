import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.environments.FourRooms import FourRooms
from src.agents.QLearning import QLearning
from src.info_theory import directed_info_approx_markov


def run(env, agent, num_episodes):
    returns = []
    obs_seqs = []
    action_seqs = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
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

        # Log episode data
        returns.append(total_reward)
        obs_seqs.append(obs_seq)
        action_seqs.append(action_seq)

    return returns, obs_seqs, action_seqs

# Moving average to smooth the curve (optional)
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


if __name__ == "__main__":
    # Hyperparameters
    NUM_EPISODES = 500
    MAX_STEPS = 100
    NUM_RUNS = 10
    ALPHA = 0.1
    GAMMA = 0.99
    EPSILON = 0.1
    MARKOV_Ks = [2, 5, 10]
    MEASURE_LENGTH = 18 # Optimal episode length for FourRooms

    all_returns = []
    all_obs_seqs = []
    all_action_seqs = []
    emp_per_episode = {k: [] for k in MARKOV_Ks}
    plast_per_episode = {k: [] for k in MARKOV_Ks}
    avg_episode_lengths = {k: [] for k in MARKOV_Ks}

    for _ in tqdm(range(NUM_RUNS)):
        env = FourRooms(max_steps=MAX_STEPS)
        agent = QLearning(
            n_actions=env.action_space.n,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon=EPSILON
        )
        returns, obs_seqs, action_seqs = run(env, agent, NUM_EPISODES)
        all_returns.append(returns)
        all_obs_seqs.append(obs_seqs)
        all_action_seqs.append(action_seqs)

    for k in MARKOV_Ks:
        for ep in range(NUM_EPISODES):
            o_seqs = [all_obs_seqs[run][ep][:MEASURE_LENGTH] for run in range(NUM_RUNS)]
            a_seqs = [all_action_seqs[run][ep][:MEASURE_LENGTH] for run in range(NUM_RUNS)]
            emp = directed_info_approx_markov(a_seqs, o_seqs, k=k)
            plast = directed_info_approx_markov(o_seqs, a_seqs, k=k)

            emp_per_episode[k].append(emp)
            plast_per_episode[k].append(plast)
            episode_lengths = [len(all_obs_seqs[run][ep]) for run in range(NUM_RUNS)]
            avg_episode_lengths[k].append(np.mean(episode_lengths))

    # Convert to array: shape = (NUM_RUNS, NUM_EPISODES)
    all_returns = np.array(all_returns)
    mean_returns = np.mean(all_returns, axis=0)
    stderr = np.std(all_returns, axis=0) / np.sqrt(NUM_RUNS)

    # Plot empowerment/plasticity
    fig, axs = plt.subplots(1, len(MARKOV_Ks)+1, figsize=(12, 3), dpi=200)
    for i, k in enumerate(MARKOV_Ks):
        axs[i].plot(emp_per_episode[k], label=f"Empowerment", color='darkorange')
        axs[i].plot(plast_per_episode[k], label=f"Plasticity", color='steelblue')
        axs[i].set_ylabel(f"Empowerment/Plasticity (k={k}) [Bits]")
        axs[i].set_xlabel("Episode")
        axs[i].set_ylim(-1, 16)

    # Plot mean return
    axs[len(MARKOV_Ks)].plot(mean_returns, label=f"Mean Return", color='black')
    axs[len(MARKOV_Ks)].set_ylabel("Mean Return")
    axs[len(MARKOV_Ks)].set_xlabel("Episode")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(wspace=0.75)
    plt.tight_layout()
    plt.show()