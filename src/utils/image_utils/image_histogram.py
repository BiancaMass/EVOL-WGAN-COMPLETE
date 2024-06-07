import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def load_images_from_folder(folder_path):
    transform = transforms.ToTensor()
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):  # Add more extensions if needed
            img = Image.open(os.path.join(folder_path, filename))
            images.append(transform(img))
    return images


def plot_histogram(images, title, ax):
    for img_tensor in images:
        img_array = img_tensor.numpy().flatten()
        ax.hist(img_array, bins=256, range=(0, 1), alpha=0.5, label='_nolegend_')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')


# Folder paths
real_image_folder = '/Volumes/SANDISK32_2/benchmarking/fid_test/real/real_20'
pqwgan_image_folder = '/Volumes/SANDISK32_2/benchmarking/fid_test/pqwgan/pqwgan_20'
classic_image_folder = '/Volumes/SANDISK32_2/benchmarking/fid_test/wgangp/fake_20'
l_04_image_folder = '/Volumes/SANDISK32_2/benchmarking/fid_test/F_04/F_04_20'
l_04_02_image_folder = '/Volumes/SANDISK32_2/benchmarking/fid_test/F_04_L2/F_04_L2_20'
l_04_03_image_folder = '/Volumes/SANDISK32_2/benchmarking/fid_test/F_04_L3/F_04_L3_20'

# Load images
real_images = load_images_from_folder(real_image_folder)
pqwgan_images = load_images_from_folder(pqwgan_image_folder)
classic_images = load_images_from_folder(classic_image_folder)
l_04_images = load_images_from_folder(l_04_image_folder)
l_04_02_images = load_images_from_folder(l_04_02_image_folder)
l_04_03_images = load_images_from_folder(l_04_03_image_folder)

# Plot histograms in a grid
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

plot_histogram(real_images, 'Real', axes[0, 0])
plot_histogram(pqwgan_images, 'PQWGAN', axes[0, 1])
plot_histogram(classic_images, 'Classic', axes[0, 2])
plot_histogram(l_04_images, 'L_04', axes[1, 0])
plot_histogram(l_04_02_images, 'L_04_L2', axes[1, 1])
plot_histogram(l_04_03_images, 'L_04_L3', axes[1, 2])

plt.tight_layout()
plt.savefig('/Users/bmassacci/Desktop/20_images_histograms.png')
