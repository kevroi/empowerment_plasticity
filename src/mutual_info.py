import numpy as np
from collections import defaultdict, Counter
from typing import Sequence, Hashable
from scipy.stats import entropy as scipy_entropy

def entropy(xs: Sequence[Hashable], mm_correction: bool = False) -> float:
    """
    H(X) = - \sum_{x \in X} p(x) \log_2 p(x)
    """
    counts = Counter(xs)
    total = len(xs)
    probabilities = np.array([count / total for count in counts.values()])
    mle_entropy = -np.sum(p * np.log2(p) for p in probabilities if p > 0)
    
    if mm_correction:
        correction = (len(counts) - 1) / (2 * total)
        # return mle_entropy - np.log2(total)
        return mle_entropy + correction
    else:
        return mle_entropy

def entropy_seq(x_seqs: Sequence[Sequence[Hashable]], k: int = 2) -> float:
    """
    Estimate H(X^n) ≈ sum_t H(X_t | X_{t-k:t-1})
    using Markov order-k approximation.

    Note:
    I initially tried to estimate the entropy of the full joint distribution
    over all possible sequences of length n (a 2^100 sized alphabet in the Bernoulli
    case), but I was in the incredibly undersampled regime, even with the MM correction.
    (In the bernoulli case, all observed samples were unqiue and had entropy of about 10 bits)

    So I switched to this incremental approach, and added a k-length history to 
    approximate each conditional entropy. This approximates the sequence to be a Markov
    process of order k.
    
    Args:
        x_seqs: list of sequences (samples) of equal length
        k: context length (history size)
    """
    seq_len = len(x_seqs[0])

    total_entropy = 0.0
    for t in range(seq_len):
        targets = []
        contexts = []
        for seq in x_seqs:
            if t < k:
                # Use empty or truncated context for early timesteps
                ctx = tuple(seq[:t])
            else:
                ctx = tuple(seq[t - k:t])
            x_t = seq[t]
            targets.append(x_t)
            contexts.append(ctx)

        h_t = conditional_entropy_from_context(targets, contexts)
        total_entropy += h_t

    return total_entropy

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

def conditional_entropy_from_context(
    targets: Sequence[Hashable],
    contexts: Sequence[tuple[Hashable]]
) -> float:
    """
    Estimate H(targets | contexts) from empirical counts.
    """
    assert len(targets) == len(contexts)
    context_to_targets = defaultdict(list)
    for ctx, x in zip(contexts, targets):
        context_to_targets[ctx].append(x)

    total = len(targets)
    cond_entropy = 0.0

    for ctx, xs in context_to_targets.items():
        count = len(xs)
        probs = np.array(list(Counter(xs).values())) / count
        h_x_given_ctx = -np.sum(p * np.log2(p) for p in probs if p > 0)
        cond_entropy += (count / total) * h_x_given_ctx

    return cond_entropy

def conditional_mutual_info(xs: Sequence[Hashable],
                            ys: Sequence[Hashable],
                            given_zs: Sequence[Hashable]) -> float:
    h_x_given_z = conditional_entropy(xs, given_zs)
    yz_pairs = list(zip(ys, given_zs))
    h_x_given_yz = conditional_entropy(xs, yz_pairs)
    return h_x_given_z - h_x_given_yz

# TO TRY:
# Masey 1990 also suggests another way to compute DI:
# I(X^n -> Y^n) = H(Y^N) - \sum{n=1}^N H(Y_n | X_n)

# directed_info / history_length for bit per step??

# empowerment has a max over DIs in its defintion

# generalised directed info - this conditions on two sequences

# try finding examples where empoweremnt and plasticyt should vary in specific ways.

# Also DI is not symmetric, so yoiu could have some test for that

# DI can never go down right? Windowed DI may.
