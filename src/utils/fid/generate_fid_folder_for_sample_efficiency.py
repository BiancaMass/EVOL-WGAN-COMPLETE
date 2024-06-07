import os
import shutil
from random import sample


def create_sample_sets(input_directory, output_directory, sample_sizes, prefix):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Get all image filenames in the input directory
    all_images = os.listdir(input_directory)
    # Exclude hidden files
    all_images = [f for f in all_images if not f.startswith('._')]
    # Ensure there are enough images in the directory
    if len(all_images) < max(sample_sizes):
        raise ValueError(f"Input directory {input_directory} contains fewer than {max(sample_sizes)} images.")

    for size in sample_sizes:
        print(f"Generating folder of size {size}")
        # Create a new subdirectory for each sample size
        size_dir = os.path.join(output_directory, f"{prefix}_{size}")
        os.makedirs(size_dir, exist_ok=True)

        # Randomly sample the specified number of images
        sampled_images = sample(all_images, size)

        for i, img in enumerate(sampled_images):
            # Copy each sampled image to the new directory with the specified naming convention
            src = os.path.join(input_directory, img)
            dst = os.path.join(size_dir, img)
            shutil.copy(src, dst)


# Define the output directories
output_dir = "/Volumes/SANDISK32_2/benchmarking/fid_test/F_04/"

# Define the input directories
input_dir = "/Volumes/SANDISK32_2/final_round5/24_05_18_21_36_23/fid_images/fake"
prefix = "F_04"
# fake_dir = os.path.join(output_dir, "fake_200")

# Define the sample sizes
sample_sizes = [200, 180, 160, 140, 120, 100, 80, 60, 40, 20]

# Create the sample sets from the input directory
create_sample_sets(input_dir, output_dir, sample_sizes, prefix=prefix)
