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


sizes = [2**i for i in range(1, 10)]
H_xs = []
H_ys = []
H_xys = []
M_xys = []
C_xys = []
CM_xyzs = []

for n in sizes:
    xs = np.random.randint(0, 2, size=n)
    ys = np.random.randint(0, 2, size=n)
    zs = np.random.randint(0, 2, size=n)

    # assert len(xs) == len(ys) == len(zs)

    H_xs.append(entropy(xs))
    H_ys.append(entropy(ys))
    H_xys.append(joint_entropy(xs, ys))
    M_xys.append(mutual_info(xs, ys))
    C_xys.append(conditional_entropy(xs, ys))
    CM_xyzs.append(conditional_mutual_info(xs, ys, zs))

    # assert abs(mutual_info(xs, ys) - mutual_info(ys, xs)) < 1e-10 # symmetry test
    # assert conditional_entropy(xs, ys) <= entropy(xs) + 1e-10
    # assert conditional_mutual_info(xs, ys, zs) >= -1e-10


# Three subplots
fig, axs = plt.subplots(6, 1)
axs[0].set_ylim(-0.1, 1.1)
axs[1].set_ylim(-0.1, 1.1)
axs[2].set_ylim(-0.1, 2.1)
axs[3].set_ylim(-0.1, 1.1)
axs[4].set_ylim(-0.1, 1.1)
axs[5].set_ylim(-0.1, 1.1)

axs[0].plot(sizes, H_xs)
axs[0].set_title('$X, Y \sim \mathrm{Bernoulli}(0.5)$')
axs[0].set_xscale('log')
axs[0].set_ylabel('$\mathbb{H}(X)$')
axs[0].axhline(y=np.log2(2), color='r', linestyle='--')

axs[1].plot(sizes, H_ys)
axs[1].set_xscale('log')
axs[1].set_ylabel('$\mathbb{H}(Y)$')
axs[1].axhline(y=np.log2(2), color='r', linestyle='--')

axs[2].plot(sizes, H_xys)
axs[2].set_xscale('log')
axs[2].set_ylabel('$\mathbb{H}(X,Y)$')
axs[2].axhline(y=np.log2(4), color='r', linestyle='--')
axs[2].set_xlabel('Sample Size')

axs[3].plot(sizes, M_xys)
axs[3].set_xscale('log')
axs[3].set_ylabel('$\mathbb{I}(X;Y)$')
axs[3].axhline(y=0, color='r', linestyle='--')
axs[3].set_xlabel('Sample Size')

axs[4].plot(sizes, C_xys)
axs[4].set_xscale('log')
axs[4].set_ylabel('$\mathbb{H}(X|Y)$')
axs[4].axhline(y=np.log2(2), color='r', linestyle='--')
axs[4].set_xlabel('Sample Size')

axs[5].plot(sizes, CM_xyzs)
axs[5].set_xscale('log')
axs[5].set_ylabel('$\mathbb{I}(X;Y|Z)$')
axs[5].axhline(y=0, color='r', linestyle='--')
axs[5].set_xlabel('Sample Size')

plt.show()








