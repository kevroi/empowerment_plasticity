# from src.mutual_info import entropy, joint_entropy, mutual_info, conditional_entropy, conditional_mutual_info

import src.mutual_info as mi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from collections import defaultdict

def drv_quantities():
    """
    Compute the important info theory quantities for a pair of discrete random variables.
    X, Y, Z ~ Bernoulli(0.5)

    We will plot the following quantities as a function of the sample size:
    - H(X)
    - H(Y)
    - H(X,Y)
    - I(X;Y)
    - H(X|Y)
    - I(X;Y|Z)
    """
    sizes = [2**i for i in range(1, 10)]
    H_xs = []
    H_ys = []
    H_xys = []
    M_xys = []
    C_xys = []
    CM_xyzs = []

    for n in tqdm(sizes,desc='Sample Sizes'):
        xs = np.random.randint(0, 2, size=n)
        ys = np.random.randint(0, 2, size=n)
        zs = np.random.randint(0, 2, size=n)

        # assert len(xs) == len(ys) == len(zs)

        H_xs.append(mi.entropy(xs))
        H_ys.append(mi.entropy(ys, mm_correction=True))
        H_xys.append(mi.joint_entropy(xs, ys))
        M_xys.append(mi.mutual_info(xs, ys))
        C_xys.append(mi.conditional_entropy(xs, ys))
        CM_xyzs.append(mi.conditional_mutual_info(xs, ys, zs))

        # assert abs(mutual_info(xs, ys) - mutual_info(ys, xs)) < 1e-10 # symmetry test
        # assert conditional_entropy(xs, ys) <= entropy(xs) + 1e-10
        # assert conditional_mutual_info(xs, ys, zs) >= -1e-10


    #  Subplots
    fig, axs = plt.subplots(3, 2, figsize=(4, 4), dpi=200)
    fig.suptitle(r'$X, Y \sim \mathrm{Bernoulli}(0.5)$')
    axs[0, 0].set_ylim(-0.1, 1.1)
    axs[0, 1].set_ylim(-0.1, 1.1)
    axs[1, 0].set_ylim(-0.1, 2.1)
    axs[1, 1].set_ylim(-0.1, 1.1)
    axs[2, 0].set_ylim(-0.1, 1.1)
    axs[2, 1].set_ylim(-0.1, 1.1)
    plt.subplots_adjust(wspace=0.5)

    axs[0, 0].plot(sizes, H_xs)
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_ylabel(r'$\mathrm{\mathbb{H}}(X)$')
    axs[0, 0].axhline(y=np.log2(2), color='r', linestyle='--')

    axs[0, 1].plot(sizes, H_ys)
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_ylabel(r'$\mathrm{\mathbb{H}}(Y)$')
    axs[0, 1].axhline(y=np.log2(2), color='r', linestyle='--')

    axs[1, 0].plot(sizes, H_xys)
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_ylabel(r'$\mathrm{\mathbb{H}}(X,Y)$')
    axs[1, 0].axhline(y=np.log2(4), color='r', linestyle='--')

    axs[1, 1].plot(sizes, M_xys)
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_ylabel(r'$\mathrm{\mathbb{I}}(X;Y)$')
    axs[1, 1].axhline(y=0, color='r', linestyle='--')

    axs[2, 0].plot(sizes, C_xys)
    axs[2, 0].set_xscale('log')
    axs[2, 0].set_ylabel(r'$\mathrm{\mathbb{H}}(X|Y)$')
    axs[2, 0].axhline(y=np.log2(2), color='r', linestyle='--')

    axs[2, 1].plot(sizes, CM_xyzs)
    axs[2, 1].set_xscale('log')
    axs[2, 1].set_ylabel(r'$\mathrm{\mathbb{I}}(X;Y|Z)$')
    axs[2, 1].axhline(y=0, color='r', linestyle='--')

    # # Hide x-axis ticks and labels for all except the bottom plot
    # for i in range(5):  # 0 to 4 (all except the last one)
    #     axs[i].tick_params(bottom=False, labelbottom=False)
    axs[2, 0].set_xlabel('Sample Size')
    axs[2, 1].set_xlabel('Sample Size')
    # Turn off ticks for all subplots except bottom row
    for i in range(3):  # rows
        for j in range(2):  # columns
            if i != 2:  # Not the bottom row
                axs[i, j].tick_params(bottom=False, labelbottom=False)
    plt.show()

def drs_quantities():
    """
    """
    history_length = 1000
    sizes = [2**i for i in range(1, 11)]
    # sizes = [100]
    H_xseqs = []
    H_yseqs = []
    D_xseq_yseqs = []

    for n in tqdm(sizes):
        x_seqs = [tuple(np.random.randint(0, 2, size=history_length)) for _ in range(n)]
        y_seqs = [tuple(np.random.randint(0, 2, size=history_length)) for _ in range(n)]

        H_xseqs.append(mi.entropy_seq(x_seqs))
        H_yseqs.append(mi.entropy_seq(y_seqs))
        D_xseq_yseqs.append(mi.directed_info_approx_markov(x_seqs, y_seqs)) # TODO incerasing k should make this go down, no?
        # D_xseq_yseqs.append(mi.directed_info_masey(x_seqs, y_seqs))

    # print(H_xseqs)
    print(D_xseq_yseqs) # TODO this goes up linearly when history length is 100
    
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(sizes, H_xseqs)
    axs[0].axhline(y=history_length, color='r', linestyle='--')
    axs[0].set_xscale('log')
    axs[0].set_ylabel(r'$\mathrm{\mathbb{H}}(X^n)$')
    axs[0].set_xlabel('Sample Size')

    axs[1].plot(sizes, D_xseq_yseqs)
    axs[1].axhline(y=0, color='r', linestyle='--')
    axs[1].set_xscale('log')
    axs[1].set_ylabel(r'$\mathrm{\mathbb{I}}(X^n -> Y^n)$')
    axs[1].set_xlabel('Sample Size')

    plt.show()


def sweep_k():
    """
    Plot the directed information as a function of sample size for different Markov orders.
    """
    history_length = 100
    # sample_sizes = [2**i for i in range(1, 17)]
    sample_sizes = [2**i for i in range(1, 14)]
    D_xseq_yseqs = defaultdict(list)
    ks = [2**i for i in range(6)]

    for k in tqdm(ks):
        for sample_size in sample_sizes:
            x_seqs = [tuple(np.random.randint(0, 2, size=history_length)) for _ in range(sample_size)]
            y_seqs = [tuple(np.random.randint(0, 2, size=history_length)) for _ in range(sample_size)]
            D_xseq_yseqs[k].append(mi.directed_info_approx_markov(x_seqs, x_seqs, k))

    colors = cm.Blues(np.linspace(0, 1, len(ks)))
    for i, (k, D) in enumerate(zip(ks, D_xseq_yseqs.values())):
        plt.plot(sample_sizes, D, label=f'k={k}', color=colors[i])
    plt.legend(title='Markov Order')
    plt.xscale('log')
    plt.ylabel(r'$\mathrm{\mathbb{I}}(X^n → Y^n)$')
    plt.xlabel('Sample Size')
    plt.axhline(y=0, color='r', linestyle='--')

    # k_peaks = [2**(2*i-1) for i in range(1,6)]
    # for k, sample_size in zip(ks[:-2], sample_sizes[:-2]):
    #     peak = (2**(2*k+2)-1)**2/(2*sample_size*np.log(2)) * (history_length%k)
    #     plt.axhline(y=peak, color='k', linestyle='--')

    plt.show()


if __name__ == "__main__":
    # drv_quantities()
    # drs_quantities()
    sweep_k()







