import jax
import jax.numpy as jnp
import numpy as np
from collections import Counter
import time

# Your optimized version
def entropy_seqs_numpy(x_seqs):
    counts = Counter(x_seqs)
    probabilities = np.array(list(counts.values())) / len(x_seqs)
    return -np.sum(probabilities * np.log2(probabilities))

# JAX version (only the final calculation)
@jax.jit
def entropy_from_probs_jax(probabilities):
    return -jnp.sum(probabilities * jnp.log2(probabilities))

def entropy_seqs_jax(x_seqs):
    counts = Counter(x_seqs)  # Still pure Python - can't avoid this
    probabilities = jnp.array(list(counts.values())) / len(x_seqs)
    return entropy_from_probs_jax(probabilities)

# Test
x_seqs = [tuple(np.random.randint(0, 2, size=10)) for _ in range(10000)]

# Benchmark
start = time.time()
result1 = entropy_seqs_numpy(x_seqs)
time1 = time.time() - start

start = time.time()
result2 = entropy_seqs_jax(x_seqs)  # Includes JIT compilation on first call
time2 = time.time() - start

print(f"NumPy: {result1:.4f} in {time1*1000:.2f}ms")
print(f"JAX: {result2:.4f} in {time2*1000:.2f}ms")