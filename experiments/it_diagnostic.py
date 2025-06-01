# from src.mutual_info import entropy, joint_entropy, mutual_info, conditional_entropy, conditional_mutual_info

import src.mutual_info as mi
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def drv_quantities():
    """
    Compute the important info theory quantities for a pair of discrete random variables.
    X, Y, Z \sim \mathrm{Bernoulli}(0.5)

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

    for n in tqdm(sizes):
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
    axs[0].set_ylabel('$\mathrm{\mathbb{H}}(X)$')
    axs[0].axhline(y=np.log2(2), color='r', linestyle='--')

    axs[1].plot(sizes, H_ys)
    axs[1].set_xscale('log')
    axs[1].set_ylabel('$\mathrm{\mathbb{H}}(Y)$')
    axs[1].axhline(y=np.log2(2), color='r', linestyle='--')

    axs[2].plot(sizes, H_xys)
    axs[2].set_xscale('log')
    axs[2].set_ylabel('$\mathrm{\mathbb{H}}(X,Y)$')
    axs[2].axhline(y=np.log2(4), color='r', linestyle='--')
    axs[2].set_xlabel('Sample Size')

    axs[3].plot(sizes, M_xys)
    axs[3].set_xscale('log')
    axs[3].set_ylabel('$\mathrm{\mathbb{I}}(X;Y)$')
    axs[3].axhline(y=0, color='r', linestyle='--')
    axs[3].set_xlabel('Sample Size')

    axs[4].plot(sizes, C_xys)
    axs[4].set_xscale('log')
    axs[4].set_ylabel('$\mathrm{\mathbb{H}}(X|Y)$')
    axs[4].axhline(y=np.log2(2), color='r', linestyle='--')
    axs[4].set_xlabel('Sample Size')

    axs[5].plot(sizes, CM_xyzs)
    axs[5].set_xscale('log')
    axs[5].set_ylabel('$\mathrm{\mathbb{I}}(X;Y|Z)$')
    axs[5].axhline(y=0, color='r', linestyle='--')
    axs[5].set_xlabel('Sample Size')

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
        # y_seqs = [tuple(np.random.randint(0, 2, size=history_length)) for _ in range(n)]

        H_xseqs.append(mi.entropy_seq(x_seqs))
        # H_yseqs.append(mi.entropy_seqs(y_seqs))
        # D_xseq_yseqs.append(mi.directed_info(x_seqs, y_seqs))
        # D_xseq_yseqs.append(mi.directed_info_masey(x_seqs, y_seqs))

    print(H_xseqs)
    # print(D_xseq_yseqs) # TODO this goes up linearly when history length is 100
    
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(sizes, H_xseqs)
    axs[0].axhline(y=history_length, color='r', linestyle='--')
    # axs[0].plot(sizes, H_yseqs)
    axs[0].set_xscale('log')
    axs[0].set_ylabel('$\mathrm{\mathbb{H}}(X^n)$')
    axs[0].set_xlabel('Sample Size')
    plt.show()

    # fig, axs = plt.subplots(2, 1)
    # # axs[0].set_ylim(-0.1, 1.1)

    # axs[0].plot(sizes, D_xseq_yseqs)
    # axs[0].set_xscale('log')
    # # axs[0].set_ylabel('$\mathbb{I}(X^n \rightarrow Y^n)$')
    # axs[0].axhline(y=0, color='r', linestyle='--')
    # axs[0].set_xlabel('Sample Size')

    # plt.show()

if __name__ == "__main__":
    # drv_quantities()
    drs_quantities()







