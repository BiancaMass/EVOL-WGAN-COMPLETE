import os
import numpy as np
from PIL import Image
from scipy.stats import wasserstein_distance

def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = Image.open(os.path.join(directory, filename)).convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize to 28x28 if necessary
            img_array = np.array(img).flatten()  # Flatten the image
            images.append(img_array)
    return np.array(images)

def normalize_images(images, mean=0.5, std=0.5):
    # Normalize images to have the specified mean and standard deviation
    images = images / 255.0  # Scale pixel values to [0, 1]  # (should not be necessary with my data)
    images = (images - np.mean(images)) / np.std(images)  # Standardize to zero mean and unit variance
    images = images * std + mean  # Rescale to desired mean and std
    return images


base_real_images_dir = "/Volumes/SANDISK32_2/benchmarking/fid_test/real/"
base_fake_images_dir = "/Volumes/SANDISK32_2/benchmarking/fid_test/F_04_L3"

real_subfolders = [name for name in os.listdir(base_real_images_dir) if os.path.isdir(os.path.join(base_real_images_dir, name))]
fake_subfolders = [name for name in os.listdir(base_fake_images_dir) if os.path.isdir(os.path.join(base_fake_images_dir, name))]

# Filter subfolders to match the pattern "real_x" and "fake_x"
real_subfolders = sorted([folder for folder in real_subfolders if folder.startswith("real_")])
fake_subfolders = sorted([folder for folder in fake_subfolders if folder.startswith("F_04_L3")])

for real_folder, fake_folder in zip(real_subfolders, fake_subfolders):
    real_images_dir = os.path.join(base_real_images_dir, real_folder)
    fake_images_dir = os.path.join(base_fake_images_dir, fake_folder)

    real_images = load_images_from_directory(real_images_dir)
    fake_images = load_images_from_directory(fake_images_dir)

    real_images = normalize_images(real_images)
    fake_images = normalize_images(fake_images)

    if len(real_images) != len(fake_images):
        print(f"Skipping {real_folder} and {fake_folder} due to unequal number of images.")
        continue

    emd = wasserstein_distance(real_images.flatten(), fake_images.flatten())
    folder_name = real_folder.split('_')[1]  # Extract number from folder name

    print(f"EMD {folder_name}: {emd}")
