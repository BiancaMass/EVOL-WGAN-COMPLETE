import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt


def wasserstein_distance_nd(data1, data2):
    return wasserstein_distance(data1.ravel(), data2.ravel())


def generate_noisy_images(base_image, noise_levels):
    noisy_images = []
    for level in noise_levels:
        noisy_image = base_image + np.random.normal(0, level, base_image.shape)
        noisy_images.append(noisy_image)
    return noisy_images

# Base image
image_size = 64
base_image = np.random.rand(image_size, image_size)

# Noise levels
noise_levels = np.linspace(0, 0.1, num=10)  # Gradual increase in noise level

# Generate noisy images
noisy_images = generate_noisy_images(base_image, noise_levels)

# Plotting and calculating Wasserstein distances
distances = []
for noisy_image in noisy_images:
    dist = wasserstein_distance_nd(base_image, noisy_image)
    distances.append(dist)

plt.plot(noise_levels, distances)
plt.xlabel('Noise Level')
plt.ylabel('Wasserstein Distance')
plt.title('Wasserstein Distance vs. Noise Level')
plt.show()
