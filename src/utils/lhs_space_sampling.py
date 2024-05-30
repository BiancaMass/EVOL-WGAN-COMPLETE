import numpy as np
from pyDOE import lhs

"""Samples for the weight experiments"""

# Generate initial samples using Latin Hypercube Sampling
n_samples = 4  # Number of initial experiments
lhs_samples = lhs(4, samples=n_samples)

# Scale samples to sum to 100
initial_weights = []
for sample in lhs_samples:
    scaled_sample = (sample / np.sum(sample)) * 100
    initial_weights.append(scaled_sample)

for weight_combo in initial_weights:
    print(weight_combo)

