from src.environments.BitWorld import BitWorld
from src.info_theory import directed_info_approx_markov
from tqdm import trange

def run_sim(agent_func, env, n_steps, n_episodes):
    """
    Runs the agent-environment loop for n_episodes episodes, each of length n_steps.

    Returns:
        act_seqs: list of sequences of actions
        obs_seqs: list of sequences of observations
    """
    act_seqs = []
    obs_seqs = []

    for _ in trange(n_episodes):
        # this section handles t=0
        obs, _ = env.reset() # O_0
        obs_seq = [obs]
        act = 0 # A_0
        act_seq = [act]

        for t in range(n_steps):
            obs_, _, _, _, _ = env.step(act)
            act_ = agent_func(t, obs, act)
            obs_seq.append(obs_)
            act_seq.append(act_)
            obs, act = obs_, act_

        obs_seqs.append(obs_seq)
        act_seqs.append(act_seq)

    return act_seqs, obs_seqs

def agent_follow_obs(t, obs, last_action):
    return obs  # A_{t+1} = O_t

if __name__ == "__main__":
    env = BitWorld()
    n_steps = 20
    n_episodes = 10000

    actions, observations = run_sim(agent_follow_obs, env, n_steps, n_episodes)

    print("Computing Empowerment (A -> O)...")
    emp = directed_info_approx_markov(actions, observations, k=2)
    print(f"Estimated Empowerment: {emp:.4f} bits")

    print("Computing Plasticity (O -> A)...")
    plast = directed_info_approx_markov(observations, actions, k=2)
    print(f"Estimated Plasticity: {plast:.4f} bits")
