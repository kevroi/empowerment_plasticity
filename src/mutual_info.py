# Becoming familiar with entropy calculations

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Sequence, Hashable

def entropy(xs: Sequence) -> float:
    counts = Counter(xs)
    total = len(xs)
    probabilities = np.array([count / total for count in counts.values()])
    return -np.sum(p * np.log2(p) for p in probabilities if p > 0)

def joint_entropy(xs: Sequence, ys: Sequence) -> float:
    joint_counts = Counter(zip(xs, ys))
    total = len(xs)
    probabilities = np.array([count / total for count in joint_counts.values()])
    return -np.sum(p * np.log2(p) for p in probabilities if p > 0)

def mutual_info(xs: Sequence, ys: Sequence) -> float:
    return entropy(xs) + entropy(ys) - joint_entropy(xs, ys)

def conditional_entropy(xs: Sequence, given_ys: Sequence) -> float:
    return joint_entropy(xs, given_ys) - entropy(given_ys)

def conditional_mutual_info(xs: Sequence, ys: Sequence, given_zs: Sequence) -> float:
    h_x_given_z = conditional_entropy(xs, given_zs)
    yz_pairs = list(zip(ys, given_zs))
    h_x_given_yz = conditional_entropy(xs, yz_pairs)
    return h_x_given_z - h_x_given_yz