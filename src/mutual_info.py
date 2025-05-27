import numpy as np
from collections import Counter
from typing import Sequence, Hashable

def entropy(xs: Sequence[Hashable]) -> float:
    counts = Counter(xs)
    total = len(xs)
    probabilities = np.array([count / total for count in counts.values()])
    return -np.sum(p * np.log2(p) for p in probabilities if p > 0)

def joint_entropy(xs: Sequence[Hashable],
                  ys: Sequence[Hashable]) -> float:
    joint_counts = Counter(zip(xs, ys))
    total = len(xs)
    probabilities = np.array([count / total for count in joint_counts.values()])
    return -np.sum(p * np.log2(p) for p in probabilities if p > 0)

def mutual_info(xs: Sequence[Hashable],
                ys: Sequence[Hashable]) -> float:
    return entropy(xs) + entropy(ys) - joint_entropy(xs, ys)

def conditional_entropy(xs: Sequence[Hashable],
                        given_ys: Sequence[Hashable]) -> float:
    return joint_entropy(xs, given_ys) - entropy(given_ys)

def conditional_mutual_info(xs: Sequence[Hashable],
                            ys: Sequence[Hashable],
                            given_zs: Sequence[Hashable]) -> float:
    h_x_given_z = conditional_entropy(xs, given_zs)
    yz_pairs = list(zip(ys, given_zs))
    h_x_given_yz = conditional_entropy(xs, yz_pairs)
    return h_x_given_z - h_x_given_yz

def directed_info(x_seqs: Sequence[Sequence[Hashable]],
                  y_seqs: Sequence[Sequence[Hashable]]) -> float:
    T = len(x_seqs[0])
    total_di = 0
    for i in range(T):
        x_pasts = [tuple(x_seq[:i+1]) for x_seq in x_seqs]
        y_nows = [y_seq[i] for y_seq in y_seqs]
        y_pasts = [tuple(y_seq[:i]) for y_seq in y_seqs]
        total_di += conditional_mutual_info(x_pasts, y_nows, y_pasts)
    return total_di

# TO TRY:
# windowed directed info
# directed_info / history_length
# empowerment has a max over DIs in its defintion
# generalised directed info - this conditions on two sequences
# try finding examples where empoweremnt and plasticyt should vary in specific ways.
# Also DI is not symmetric, so yoiu could have some test for that
# DI can never go down right? Windowed DI may.
