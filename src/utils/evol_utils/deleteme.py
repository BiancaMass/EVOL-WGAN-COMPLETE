import numpy as np
import scipy


# Generate a 28x28 matrix with random values between -1 and 1
# Example synthetic data: two batches of 10 images, each 28x28 pixels
u = np.random.rand(10, 28, 28)
v = np.random.rand(10, 28, 28)

# Flatten each image to a 1D vector (10 images, each now a vector of 784 pixels)
u_flattened = u.reshape(10, -1)  # Reshape to have 10 rows and 784 columns
v_flattened = v.reshape(10, -1)  # Reshape similarly


distance = scipy.stats.wasserstein_distance_nd(u_values=u,
                                               v_values=v)

print(distance)