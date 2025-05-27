import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import Counter
from src.mutual_info import entropy, mutual_info, conditional_mutual_info

class DirectedInformationEstimator:
    """
    Estimates directed information I(A^n → O^n) between action and observation sequences.
    Uses a simple histogram-based estimator for the conditional mutual information terms.
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Args:
            n_bins: Number of bins to use for histogram estimation
        """
        self.n_bins = n_bins
        
    def _discretize_sequence(self, sequence: List[Tuple]) -> np.ndarray:
        """Convert a sequence of tuples into a discrete array for histogram estimation."""
        # Convert each tuple into a single integer by treating it as a base-n number
        max_vals = [max(x[i] for x in sequence) + 1 for i in range(len(sequence[0]))]
        bases = np.cumprod([1] + max_vals[:-1])
        
        return np.array([sum(x[i] * b for i, b in enumerate(bases)) for x in sequence])
    
    def _estimate_conditional_mutual_info(self, 
                                        actions: List[int], 
                                        obs: List[Tuple],
                                        t: int) -> float:
        """
        Estimate I(A[1:t]; O[t] | O[1:t-1]) using histogram-based estimation.
        """
        # Convert sequences to discrete arrays
        action_seq = np.array(actions[:t])
        obs_t = self._discretize_sequence([obs[t-1]])[0]  # Current observation
        past_obs = self._discretize_sequence(obs[:t-1]) if t > 1 else np.array([])
        
        # Create joint histogram
        if len(past_obs) == 0:
            # If no past observations, just estimate I(A[1:t]; O[t])
            hist = np.histogram2d(action_seq, [obs_t], bins=self.n_bins)[0]
            p_joint = hist / (hist.sum() + 1e-10)
            p_action = p_joint.sum(axis=1)
            p_obs = p_joint.sum(axis=0)
            
            # Calculate mutual information
            mi = 0
            for i in range(len(p_action)):
                for j in range(len(p_obs)):
                    if p_joint[i,j] > 0:
                        mi += p_joint[i,j] * np.log2(p_joint[i,j] / (p_action[i] * p_obs[j] + 1e-10))
            return mi
        else:
            # For conditional mutual information, we'll use a simpler approach
            # by binning the action sequence and observations separately
            action_bins = np.linspace(action_seq.min() - 0.5, action_seq.max() + 0.5, self.n_bins + 1)
            obs_bins = np.linspace(min(past_obs.min(), obs_t) - 0.5, 
                                 max(past_obs.max(), obs_t) + 0.5, 
                                 self.n_bins + 1)
            
            # Create 2D histograms for each pair, ensuring same length
            # For action-past_obs, we need to match lengths
            min_len = min(len(action_seq), len(past_obs))
            hist_action_past = np.histogram2d(action_seq[:min_len], past_obs[:min_len], 
                                            bins=[action_bins, obs_bins])[0]
            
            # For obs_t-past_obs, we need to repeat obs_t
            obs_t_repeated = np.full_like(past_obs, obs_t)
            hist_obs_past = np.histogram2d(obs_t_repeated, past_obs, 
                                         bins=[obs_bins, obs_bins])[0]
            
            # For action-obs_t, we need to repeat obs_t
            obs_t_for_action = np.full_like(action_seq, obs_t)
            hist_joint = np.histogram2d(action_seq, obs_t_for_action, 
                                      bins=[action_bins, obs_bins])[0]
            
            # Normalize histograms
            p_action_past = hist_action_past / (hist_action_past.sum() + 1e-10)
            p_obs_past = hist_obs_past / (hist_obs_past.sum() + 1e-10)
            p_joint = hist_joint / (hist_joint.sum() + 1e-10)
            
            # Calculate conditional mutual information
            cmi = 0
            for i in range(len(action_bins)-1):
                for j in range(len(obs_bins)-1):
                    if p_joint[i,j] > 0 and p_action_past[i,j] > 0 and p_obs_past[i,j] > 0:
                        cmi += p_joint[i,j] * np.log2(
                            (p_joint[i,j] * p_action_past[i,j]) / 
                            (p_action_past[i,j] * p_obs_past[i,j] + 1e-10)
                        )
            return cmi
    
    def estimate_directed_info(self, 
                             actions: List[int], 
                             obs: List[Tuple], 
                             window_size: Optional[int] = None) -> float:
        """
        Estimate directed information I(A[1:n] → O[1:n]) using the sum of conditional mutual information terms.
        
        Args:
            actions: List of actions
            obs: List of observations
            window_size: Optional window size to use for estimation. If None, uses full sequence.
            
        Returns:
            Estimated directed information
        """
        if window_size is None:
            window_size = len(actions)
        
        # Ensure sequences are the same length
        min_len = min(len(actions), len(obs), window_size)
        actions = actions[:min_len]
        obs = obs[:min_len]
        
        # Sum the conditional mutual information terms
        di = 0
        for t in range(1, min_len + 1):
            di += self._estimate_conditional_mutual_info(actions, obs, t)
        
        return di

def collect_room_sequences(env, room_idx: int, n_steps: int = 1000) -> Tuple[List[int], List[Tuple]]:
    """
    Collect action and observation sequences while staying in a specific room.
    
    Args:
        env: The LightRooms environment
        room_idx: Room to stay in
        n_steps: Number of steps to collect
        
    Returns:
        Tuple of (actions, observations)
    """
    actions = []
    observations = []
    
    # Reset environment
    obs, _ = env.reset()
    
    for _ in range(n_steps):
        # If not in target room, move there
        if env.current_room < room_idx:
            action = 1  # move right
        elif env.current_room > room_idx:
            action = 0  # move left
        else:
            # In target room, randomly choose between staying and pulling lever
            action = np.random.choice([1, 2], p=[0.5, 0.5])  # 50% chance to pull lever
        
        # Take step
        obs, _, _, _, _ = env.step(action)
        
        # Record action and observation
        actions.append(action)
        observations.append(obs)
    
    return actions, observations

def plot_empowerment_by_room(env, n_steps: int = 1000, n_trials: int = 5):
    """
    Plot the estimated empowerment (directed information of actions on
    observations) for each room.
    
    Args:
        env: The LightRooms environment
        n_steps: Number of steps to collect per trial
        n_trials: Number of trials to average over
    """
    estimator = DirectedInformationEstimator()
    room_empowerments = []
    
    for room in range(env.n_total_rooms):
        room_di = []
        for _ in range(n_trials):
            actions, obs = collect_room_sequences(env, room, n_steps)
            di = estimator.estimate_directed_info(actions, obs)
            room_di.append(di)
        room_empowerments.append(np.mean(room_di))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    rooms = np.arange(env.n_total_rooms)
    control_probs = rooms / env.n_rooms
    
    plt.plot(control_probs, room_empowerments, 'bo-', label='Estimated Empowerment')
    plt.plot(control_probs, control_probs, 'r--', label='Control Probability')
    
    plt.xlabel('Room Control Probability (i/n)')
    plt.ylabel('Directed Information (Empowerment)')
    plt.title('Empowerment vs Room Control Probability')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()

def empirical_mi(joint_counts, x_counts, y_counts, n):
    """
    Empirical mutual information from counts.
    """
    mi = 0.0
    for (x, y), joint in joint_counts.items():
        pxy = joint / n
        px = x_counts[x] / n
        py = y_counts[y] / n
        mi += pxy * np.log2(pxy / (px * py + 1e-12) + 1e-12)
    return mi

def empirical_cond_mi(joint_counts, xz_counts, yz_counts, z_counts, n):
    """
    Empirical conditional mutual information from counts.
    """
    cmi = 0.0
    for (x, y, z), joint in joint_counts.items():
        pxyz = joint / n
        pxz = xz_counts[(x, z)] / n
        pyz = yz_counts[(y, z)] / n
        pz = z_counts[z] / n
        cmi += pxyz * np.log2((pz * pxyz) / (pxz * pyz + 1e-12) + 1e-12)
    return cmi

def collect_short_sequences(env, room_idx: int, n_samples: int = 2000):
    """
    Collect short (A1, O1, A2, O2) sequences for a given room.
    """
    samples = []
    for _ in range(n_samples):
        obs, _ = env.reset()
        # Move to target room
        while env.current_room < room_idx:
            obs, *_ = env.step(1)
        while env.current_room > room_idx:
            obs, *_ = env.step(0)
        # Take two actions in the room (always pull lever)
        a1 = 2
        obs1, *_ = env.step(a1)
        a2 = 2
        obs2, *_ = env.step(a2)
        samples.append((a1, obs[1], a2, obs1[1], obs2[1]))
    return samples

def estimate_directed_info_short(samples: List[Tuple[int, int, int, int, int]]):
    """
    Estimate directed information for short sequences: (A1, O1, A2, O2, O3)
    Returns: I(A1; O1) + I(A1, A2; O2 | O1)
    """
    n = len(samples)
    # Unpack
    a1s = [s[0] for s in samples]
    o1s = [s[1] for s in samples]
    a2s = [s[2] for s in samples]
    o2s = [s[3] for s in samples]
    o3s = [s[4] for s in samples]
    # I(A1; O1)
    joint1 = Counter(zip(a1s, o1s))
    a1_counts = Counter(a1s)
    o1_counts = Counter(o1s)
    mi1 = empirical_mi(joint1, a1_counts, o1_counts, n)
    # I((A1, A2); O2 | O1)
    joint2 = Counter(zip(a1s, a2s, o2s, o1s))
    xz_counts = Counter(zip(a1s, a2s, o1s))
    yz_counts = Counter(zip(o2s, o1s))
    z_counts = Counter(o1s)
    cmi2 = 0.0
    for (a1, a2, o2, o1), joint in joint2.items():
        pxy_z = joint / n
        pxz = xz_counts[(a1, a2, o1)] / n
        pyz = yz_counts[(o2, o1)] / n
        pz = z_counts[o1] / n
        cmi2 += pxy_z * np.log2((pz * pxy_z) / (pxz * pyz + 1e-12) + 1e-12)
    return mi1 + cmi2

def plot_empowerment_by_room_short(env, n_samples: int = 2000):
    """
    Plot empowerment (directed info) for each room using short-sequence estimator.
    """
    room_empowerments = []
    for room in range(env.n_total_rooms):
        samples = collect_short_sequences(env, room, n_samples)
        di = estimate_directed_info_short(samples)
        room_empowerments.append(di)
    rooms = np.arange(env.n_total_rooms)
    control_probs = rooms / env.n_rooms
    plt.figure(figsize=(10, 6))
    plt.plot(control_probs, room_empowerments, 'bo-', label='Estimated Empowerment (short seq)')
    plt.plot(control_probs, control_probs, 'r--', label='Control Probability')
    plt.xlabel('Room Control Probability (i/n)')
    plt.ylabel('Directed Information (Empowerment)')
    plt.title('Empowerment vs Room Control Probability (Short Sequence)')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def get_room_short_sequences_from_trajectory(act_seq, obs_seq, room_idx):
    """
    Extract short (A1, O1, A2, O2, O3) subsequences from a long trajectory where the agent is in the given room.
    Only includes contiguous blocks of at least 3 steps in the room.
    """
    samples = []
    n = len(act_seq)
    i = 0
    while i < n - 2:
        # Check for a block of at least 3 consecutive steps in the room
        if obs_seq[i][0] == room_idx and obs_seq[i+1][0] == room_idx and obs_seq[i+2][0] == room_idx:
            a1 = act_seq[i]
            o1 = obs_seq[i][1]
            a2 = act_seq[i+1]
            o2 = obs_seq[i+1][1]
            o3 = obs_seq[i+2][1]
            samples.append((a1, o1, a2, o2, o3))
            i += 1  # Overlapping subsequences are allowed
        else:
            i += 1
    return samples

def plot_empowerment_by_room_from_long_trajectory(env, n_steps=10000):
    """
    Run a long episode, filter for each room, and estimate empowerment using short-sequence estimator.
    """
    act_seq, obs_seq = collect_long_trajectory(env, n_steps=n_steps)
    room_empowerments = []
    for room in range(env.n_total_rooms):
        samples = get_room_short_sequences_from_trajectory(act_seq, obs_seq, room)
        if len(samples) > 0:
            di = estimate_directed_info_short(samples)
        else:
            di = 0.0
        room_empowerments.append(di)
    rooms = np.arange(env.n_total_rooms)
    control_probs = rooms / env.n_rooms
    plt.figure(figsize=(10, 6))
    plt.plot(control_probs, room_empowerments, 'bo-', label='Estimated Empowerment (filtered short seq)')
    plt.plot(control_probs, control_probs, 'r--', label='Control Probability')
    plt.xlabel('Room Control Probability (i/n)')
    plt.ylabel('Directed Information (Empowerment)')
    plt.title('Empowerment vs Room Control Probability (Filtered Short Sequences)')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def collect_long_trajectory(env, n_steps=10000):
    obs_seq = []
    act_seq = []
    obs, _ = env.reset()
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        obs_seq.append(obs)
        act_seq.append(action)
    return act_seq, obs_seq

def conditional_entropy(xs, conds):
    """Empirical H(xs | conds) using your entropy code."""
    xs = np.array(xs)
    conds = np.array(conds)
    if len(xs) == 0:
        return 0.0
    if conds.ndim == 1:
        conds = conds.reshape(-1, 1)
    unique_conds, cond_indices = np.unique(conds, axis=0, return_inverse=True)
    h = 0.0
    for idx in range(len(unique_conds)):
        mask = cond_indices == idx
        if np.sum(mask) == 0:
            continue
        h += (np.sum(mask) / len(xs)) * entropy(xs[mask])
    return h

def simple_directed_info(actions, observations):
    # Use only the light state for empowerment
    obs_light = np.array([o[1] for o in observations])
    actions = np.array(actions)
    n = len(actions)
    di = 0.0
    for i in range(n):
        if i == 0:
            # I(A_0; O_0)
            di += mutual_info(actions[:1], obs_light[:1])
        else:
            # I(A^{i+1}; O_i | O^{i})
            # Xs: tuple of actions up to i (A_0,...,A_i)
            # Ys: O_i
            # Zs: tuple of previous obs (O_0,...,O_{i-1})
            Xs = [tuple(actions[j:j+i+1]) for j in range(n - i)]
            Ys = obs_light[i:]
            Zs = [tuple(obs_light[j:j+i]) for j in range(n - i)]
            di += conditional_mutual_info(Xs, Ys, Zs)
    return di

def plot_empowerment_by_room_lever_only_simple(env, n_pulls=100):
    room_empowerments = []
    for room in range(env.n_total_rooms):
        obs, _ = env.reset()
        while env.current_room < room:
            obs, *_ = env.step(1)
        while env.current_room > room:
            obs, *_ = env.step(0)
        actions = []
        observations = []
        for _ in range(n_pulls):
            # action = 2
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            actions.append(action)
            observations.append(obs)
        if len(actions) > 1:
            di = simple_directed_info(actions, observations)
        else:
            di = 0.0
        room_empowerments.append(di)
    rooms = np.arange(env.n_total_rooms)
    control_probs = rooms / env.n_rooms
    plt.figure(figsize=(10, 6))
    plt.plot(control_probs, room_empowerments, 'bo-', label='Empowerment (lever only, simple)')
    # plt.plot(control_probs, control_probs, 'r--', label='Control Probability')
    plt.xlabel('Room Control Probability (i/n)')
    plt.ylabel('Directed Information (Empowerment)')
    plt.title('Empowerment vs Room Control Probability (Lever Only, Simple)')
    plt.legend()
    plt.grid(True)
    return plt.gcf() 