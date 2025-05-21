# Becoming familiar with entropy calculations

import numpy as np
import matplotlib.pyplot as plt

def entropy(xs):
    unique, counts = np.unique(xs, return_counts=True)
    probabilities = counts / len(xs)
    return np.sum(-probabilities * np.log2(probabilities))

def joint_entropy(xs, ys):
    pairs = np.array(list(zip(xs, ys)))
    unique_pairs, counts = np.unique(pairs, return_counts=True)
    probabilities = counts / len(pairs)
    return np.sum(-probabilities * np.log2(probabilities))

def mutual_info(xs, ys):
    return entropy(xs) + entropy(ys) - joint_entropy(xs, ys)


sizes = [2**i for i in range(1, 10)]
H_xs = []
H_ys = []
H_xys = []

for n in sizes:
    xs = np.random.randint(0, 2, size=n)
    ys = np.random.randint(0, 2, size=n)
    H_xs.append(entropy(xs))
    H_ys.append(entropy(ys))
    H_xys.append(mutual_info(xs, ys))


# Three subplots
fig, axs = plt.subplots(3, 1)
axs[0].set_ylim(0, 1.1)
axs[1].set_ylim(0, 1.1)
axs[2].set_ylim(0, 2.1)

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
axs[2].set_ylabel('$\mathbb{I}(X;Y)$')
axs[2].axhline(y=np.log2(4), color='r', linestyle='--')
axs[2].set_xlabel('Sample Size')

plt.show()








