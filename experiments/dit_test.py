# Install: pip install dit
import dit
import numpy as np
from collections import Counter
from dit.shannon import entropy

# Create distribution from your sequences
# def compute_directed_info_dit(X, Y):
#     # Convert sequences to joint distribution
#     xy_pairs = list(zip(X, Y))
#     
#     # Count frequencies of each outcome
#     counts = Counter(xy_pairs)
#     total = len(xy_pairs)
#     
#     # Create distribution with probabilities
#     outcomes = list(counts.keys())
#     pmf = [count/total for count in counts.values()]
#     d = dit.Distribution(outcomes, pmf)
#     
#     # Compute transfer entropy (approximates directed information)
#     # We need to specify which variables are X and Y
#     d.set_rv_names('XY')  # Name the random variables
#     return dit.multivariate.transfer_entropy(d, 'X', 'Y')

# Example
X = [0, 1, 1, 0, 1, 0]
Y = [0, 0, 1, 1, 1, 0] 
# di = compute_directed_info_dit(X, Y)

# Example for mutual information
X = [0, 1, 1, 0, 1, 0]
Y = [0, 0, 1, 1, 1, 0]

xy_pairs = list(zip(X, Y))
counts = Counter(xy_pairs)
total = len(xy_pairs)
outcomes = list(counts.keys())
pmf = [count/total for count in counts.values()]
d = dit.Distribution(outcomes, pmf)
d.set_rv_names('XY')
from dit.shannon import mutual_information
mi = mutual_information(d, ['X'], ['Y'])
print("Mutual Information I(X;Y):", mi)

# Example for conditional mutual information
Z = [1, 0, 1, 0, 1, 0]
xyz_triples = list(zip(X, Y, Z))
counts = Counter(xyz_triples)
total = len(xyz_triples)
outcomes = list(counts.keys())
pmf = [count/total for count in counts.values()]
d = dit.Distribution(outcomes, pmf)
d.set_rv_names('XYZ')

# d is your dit.Distribution with variables named 'X', 'Y', 'Z'
h_xz = entropy(d, ['X', 'Z'])
h_yz = entropy(d, ['Y', 'Z'])
h_xyz = entropy(d, ['X', 'Y', 'Z'])
h_z = entropy(d, ['Z'])

cmi = h_xz + h_yz - h_xyz - h_z
print("Conditional Mutual Information I(X;Y|Z):", cmi)