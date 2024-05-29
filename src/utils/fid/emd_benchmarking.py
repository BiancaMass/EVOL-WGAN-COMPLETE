import os
import numpy as np
from PIL import Image
from scipy.stats import wasserstein_distance_nd


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
    images = images / 255.0  # Scale pixel values to [0, 1]
    images = (images - np.mean(images)) / np.std(images)  # Standardize to zero mean and unit variance
    images = images * std + mean  # Rescale to desired mean and std
    return images


test_images_dir = "/Volumes/SANDISK32_2/benchmarking/fid_test/real/real_200"
real_images = load_images_from_directory(test_images_dir)
real_images = normalize_images(real_images)

fake_images_base_dir = "/Volumes/SANDISK32_2/final_round6/"

fake_subfolders = [name for name in os.listdir(fake_images_base_dir) if os.path.isdir(os.path.join(fake_images_base_dir, name))]

for folder in fake_subfolders:
    print("*** CURRENT FOLDER ***")
    fake_images_dir = os.path.join(fake_images_base_dir, folder, "fid_images/fake")

    fake_images = load_images_from_directory(fake_images_dir)
    fake_images = normalize_images(fake_images)
    emd = wasserstein_distance_nd(real_images, fake_images)
    print(f"EMD for {folder}: {emd}")
