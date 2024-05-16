import os
import shutil
from random import sample


def create_sample_sets(input_dir, output_dir, sample_sizes, prefix):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all image filenames in the input directory
    all_images = os.listdir(input_dir)

    # Ensure there are enough images in the directory
    if len(all_images) < max(sample_sizes):
        raise ValueError(f"Input directory {input_dir} contains fewer than {max(sample_sizes)} images.")

    for size in sample_sizes:
        # Create a new subdirectory for each sample size
        size_dir = os.path.join(output_dir, f"{prefix}_{size}")
        os.makedirs(size_dir, exist_ok=True)

        # Randomly sample the specified number of images
        sampled_images = sample(all_images, size)

        for i, img in enumerate(sampled_images):
            # Copy each sampled image to the new directory with the specified naming convention
            src = os.path.join(input_dir, img)
            dst = os.path.join(size_dir, img)
            shutil.copy(src, dst)


# Define the output directories
output_dir = "/Volumes/SANDISK32_2/fid_test"

# Define the input directories
real_dir_0 = os.path.join(output_dir, "real_0_200")
real_dir_1 = os.path.join(output_dir, "real_1_200")



# Define the sample sizes
sample_sizes = [180, 160, 140, 120, 100, 80, 60, 40, 20]

# Create the sample sets for real images
create_sample_sets(real_dir_0, output_dir, sample_sizes, "real_0")

# Create the sample sets for generated images
create_sample_sets(real_dir_1, output_dir, sample_sizes, "real_1")
