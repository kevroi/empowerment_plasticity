from src.environments.BitWorld import BitWorld
from src.mutual_info import directed_info_approx_markov
from tqdm import trange

def run_sim(agent_func, env, n_steps, n_episodes):
    act_seqs = []
    obs_seqs = []

    for _ in trange(n_episodes):
        obs, _ = env.reset() # O_0
        obs_seq = [] # = [O_1, O_2, ... O_n]
        act = 0 # A_0
        act_seq = [] # = [A_0, A_1, ... A_{n-1}]
        act_seq.append(act)

        for t in range(n_steps):
            obs_seq.append(obs)
            act = agent_func(t, obs, act)
            act_seq.append(act)
            obs, _, _, _, _ = env.step(act)

        obs_seqs.append(obs_seq)
        act_seqs.append(act_seq[:-1])

    return act_seqs, obs_seqs

def agent_follow_obs(t, obs, last_action):
    return obs  # A_{t+1} = O_t

if __name__ == "__main__":
    env = BitWorld()
    n_steps = 20
    n_episodes = 1000

    actions, observations = run_sim(agent_follow_obs, env, n_steps, n_episodes)

    print("Computing Empowerment (A -> O)...")
    emp = directed_info_approx_markov(actions, observations, k=2)
    print(f"Estimated Empowerment: {emp:.4f} bits")

    print("Computing Plasticity (O -> A)...")
    plast = directed_info_approx_markov(observations, actions, k=2)
    print(f"Estimated Plasticity: {plast:.4f} bits")
