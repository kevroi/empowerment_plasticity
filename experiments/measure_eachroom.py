import matplotlib.pyplot as plt
import numpy as np
from src.environments.LightRooms import LightRooms
from src.info_theory import directed_info_approx_markov
from tqdm import tqdm
from collections import defaultdict


def collect_data_from_start_room(env, start_room, num_samples, seq_len, rng):
    """
    Rolls out agent-environment loop to collect action and observation sequences.

    Returns:
        action_seqs: list of sequences of actions
        obs_seqs: list of sequences of observations
    """
    action_seqs = []
    obs_seqs = []

    for _ in range(num_samples):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000))) # O_0
        env.current_room = start_room
        obs_seq = []
        act_seq = []

        for _ in range(seq_len):
            obs_seq.append(obs)
            action = env.action_space.sample() # Uniform Policy
            # action = obs_seq[-1][1] # Follow obs
            # action = 2 #  exactly zero emp and plasitcity as A^n = [2,2,2,...], O^n = [0,1,0,1,...]
            # action = np.random.randint(0, 2) # approx 0 as A^n is random but uncorrelated with O^n
            # action = env.x_state
            act_seq.append(action)
            obs, _, _, _, _ = env.step(action)

        action_seqs.append(act_seq)
        obs_seqs.append(obs_seq)

    return action_seqs, obs_seqs

def flatten_obs_seq(obs_seqs):
    # removes room number from obs
    return [[(light) for room, light in seq] for seq in obs_seqs]

def measure_emp_plast_by_room(env, n_rooms, seq_len, num_samples, markov_k, rng):
    """
    Measures empowerment and plastiicty by changing the starting room for rollouts.
    Agent is a uniform random policy.

    Returns:
        empowerments: list of empowerment values
        plasticities: list of plasticity values
    """
    empowerments = []
    plasticities = []

    for room in tqdm(range(n_rooms + 1), desc="Measuring by room index"):
        a_seqs, o_seqs = collect_data_from_start_room(env, room, num_samples, seq_len, rng)
        o_seqs = flatten_obs_seq(o_seqs)

        emp = directed_info_approx_markov(a_seqs, o_seqs, k=markov_k)
        plast = directed_info_approx_markov(o_seqs, a_seqs, k=markov_k)

        empowerments.append(emp)
        plasticities.append(plast)

    # --- Plotting ---
    room_indices = list(range(n_rooms + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(room_indices, empowerments, marker='o', label='Empowerment (A→O)')
    plt.plot(room_indices, plasticities, marker='s', label='Plasticity (O→A)')
    plt.xticks(room_indices, [int(i) for i in room_indices])
    plt.xlabel('Starting Room Index')
    plt.ylabel('Bits')
    plt.title(r'$\mathfrak{E}$ and $\mathfrak{P}$ vs. Starting Room')
    plt.legend()
    plt.tight_layout()
    plt.show()

def sweep_seq_len(n_rooms, seq_lens, num_samples, markov_k, rng):
    """
    Sweeps seq_len of rollouts used to measure empowerment and plasticty.
    Agent is a uniform random policy.

    Returns:
        Emp_by_seq_len: dict of seq_len to list of empowerment values
        Plast_by_seq_len: dict of seq_len to list of plasticity values
    """
    Emp_by_seq_len = defaultdict(list)
    Plast_by_seq_len = defaultdict(list)

    for seq_len in tqdm(seq_lens, desc="Sweeping seq len"):
        env = LightRooms(n_rooms=n_rooms)
        for room in range(n_rooms + 1):
            a_seqs, o_seqs = collect_data_from_start_room(env, room, num_samples, seq_len, rng)
            o_seqs = flatten_obs_seq(o_seqs)
            emp = directed_info_approx_markov(a_seqs, o_seqs, k=markov_k)
            plast = directed_info_approx_markov(o_seqs, a_seqs, k=markov_k)
            Emp_by_seq_len[seq_len].append(emp)
            Plast_by_seq_len[seq_len].append(plast)

    return Emp_by_seq_len, Plast_by_seq_len


if __name__ == "__main__":
    # --- Parameters ---
    N_ROOMS = 4
    SEQ_LEN = 10
    NUM_SAMPLES = 1000
    MARKOV_K = 2
    RNG = np.random.default_rng(0)

    env = LightRooms(n_rooms=N_ROOMS)

    # --- Measure Empowerment and Plasticty by room ---
    measure_emp_plast_by_room(env, N_ROOMS, SEQ_LEN, NUM_SAMPLES, MARKOV_K, RNG)

    # --- Repeat but sweep seq len we calculate DI over ---
    SEQ_LENS = [10, 100, 1000]
    Emp_by_seq_len, Plast_by_seq_len = sweep_seq_len(N_ROOMS, SEQ_LENS, NUM_SAMPLES, MARKOV_K, RNG)

    # --- Plotting ---
    room_indices = list(range(N_ROOMS + 1))
    fig, axs = plt.subplots(1, 3, figsize=(6, 1.5), dpi=200)
    for i, (T, E, P) in enumerate(zip(SEQ_LENS, Emp_by_seq_len.values(), Plast_by_seq_len.values())):
        axs[i].plot(room_indices, E, label=f'Empowerment')
        axs[i].plot(room_indices, P, label=f'Plasticity')
        axs[i].set_ylabel('Bits')
        axs[i].set_xlabel('Starting Room')
        axs[i].set_title(f'T = {T}')
        axs[i].set_xticks(list(range(N_ROOMS+1)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(wspace=0.75)
    plt.show()