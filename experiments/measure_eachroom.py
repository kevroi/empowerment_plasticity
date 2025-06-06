import matplotlib.pyplot as plt
import numpy as np
from src.environments.LightRooms import LightRooms
from src.info_theory import directed_info_approx_markov
from tqdm import tqdm
from collections import defaultdict
import matplotlib.cm as cm

# --- Parameters ---
N_ROOMS = 4
SEQ_LEN = 10
NUM_SAMPLES = 1000
MARKOV_K = 2
RNG = np.random.default_rng(0)

# --- Create environment ---
env = LightRooms(n_rooms=N_ROOMS)

# --- Data collection function ---
def collect_data_from_start_room(start_room, num_samples, seq_len, rng):
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
    # remove room number from obs
    return [[(light) for room, light in seq] for seq in obs_seqs]

def measure_emp_plast_by_room(n_rooms, seq_len, num_samples, markov_k, rng):
    empowerments = []
    plasticities = []

    for room in tqdm(range(n_rooms + 1), desc="Measuring by room index"):
        a_seqs, o_seqs = collect_data_from_start_room(room, num_samples, seq_len, rng)
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
    room_indices = list(range(n_rooms + 1))
    seq_lens = [10, 100, 1000]
    Emp_by_seq_len = defaultdict(list)
    Plast_by_seq_len = defaultdict(list)

    for seq_len in tqdm(seq_lens, desc="Sweeping seq len"):
        for room in range(n_rooms + 1):
            a_seqs, o_seqs = collect_data_from_start_room(room, num_samples, seq_len, rng)
            o_seqs = flatten_obs_seq(o_seqs)
            emp = directed_info_approx_markov(a_seqs, o_seqs, k=markov_k)
            plast = directed_info_approx_markov(o_seqs, a_seqs, k=markov_k)
            Emp_by_seq_len[seq_len].append(emp)
            Plast_by_seq_len[seq_len].append(plast)

    return Emp_by_seq_len, Plast_by_seq_len


measure_emp_plast_by_room(N_ROOMS, SEQ_LEN, NUM_SAMPLES, MARKOV_K, RNG)

# SEQ_LENS = [10, 100, 1000]
# sweep_seq_len(N_ROOMS, SEQ_LENS, NUM_SAMPLES, MARKOV_K, RNG)