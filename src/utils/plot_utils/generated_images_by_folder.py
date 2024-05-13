"""Generates an image grid of example images taken from the fid/fake sub-dir from all dirs in a folder."""

# PLAY WITH THE IMAGE SIZE FOR BEST RESULTS CAUSE IT DEPENDS ON THE TOTAL NUMBER OF IMAGES

import os
import matplotlib.pyplot as plt
from PIL import Image
import random

# Path to the main directory
main_dir = "/Volumes/SANDISK32_2/final_round3"
output_dir = "/Volumes/SANDISK32_2/plots_latex/"
image_name = "round_3_output_examples"
output_path = os.path.join(output_dir, image_name)

# Dictionary mapping folder names to titles
# For ROUND 1:
# experiment_titles = {
#     "24_04_29_13_25_46": "E_28_1",
#     "24_04_29_17_05_32": "E_28_2",
#     "24_04_29_21_41_24": "E_28_3",
#     "24_04_29_17_35_15": "E_14_1",
#     "24_04_30_07_02_51": "E_14_2",
#     "24_04_30_09_36_08": "E_14_3",
#     "24_04_29_19_39_19": "E_7_1",
#     "24_04_30_07_56_56": "E_7_2",
#     "24_04_30_13_03_42": "E_7_3",
#     "24_04_30_11_18_02": "E_2_1",
#     "24_04_30_13_07_07": "E_2_2",
#     "24_04_30_13_32_01": "E_2_3",
#     "24_04_30_17_07_37": "E_1_1",
#     "24_04_30_17_15_23": "E_1_2",
#     "24_04_30_19_48_33": "E_1_3"}

# For ROUND 2:
# experiment_titles = {
#     "24_05_07_15_41_05": "A_28-1-1-01",
#     "24_05_07_15_56_10": "A_28-1-1-02",
#     "24_05_07_19_38_37": "A_28-1-1-03",
#     "24_05_06_16_09_28": "A_28-1-2-01",
#     "24_05_09_08_03_50": "A_28-1-2-02",
#     "24_05_08_07_37_33": "A_28-1-2-03",
#     "24_05_08_06_26_09": "A_28-1-3-01",
#     "24_05_08_06_26_52": "A_28-1-3-02",
#     "24_05_08_06_27_13": "A_28-1-3-03",
#     "24_05_08_10_55_44": "A_14-1-1-01",
#     "24_05_08_13_37_23": "A_14-1-1-02",
#     "24_05_08_16_09_36": "A_14-1-1-03",
#     "24_05_08_16_26_53": "A_14-1-2-01",
#     "24_05_08_16_29_33": "A_14-1-2-02",
#     "24_05_08_16_48_47": "A_14-1-2-03",
#     "24_05_09_07_03_15": "A_14-1-3-01",
#     "24_05_09_07_03_46": "A_14-1-3-02",
#     "24_05_09_08_04_26": "A_14-1-3-03"}


# Function to randomly select images from a directory

# FOR ROUND 3:
experiment_titles = {"24_05_07_19_38_37": "A_28-1-1-03",
                     "24_05_10_08_28_38": "W_28-1-1-60-30",
                     "24_05_10_08_57_17": "W_28-1-1-65-25",
                     "24_05_10_08_58_32": "W_28-1-1-70-20",
                     "24_05_10_09_09_26": "W_28-1-1-75-15",
                     "24_05_10_14_18_28": "W_28-1-1-80-10",
                     "24_05_11_07_52_57": "W_28-1-1-85-5"}


def select_images(image_directory, num_images):
    images_subgrid = os.listdir(image_directory)
    selected_images = random.sample(images_subgrid, num_images)
    return [Image.open(os.path.join(image_directory, im)) for im in selected_images]


# valid_directories = [d for d in sorted(os.listdir(main_dir)) if not d.startswith('._')]
valid_directories = [i for i in sorted(os.listdir(main_dir)) if i in experiment_titles.keys()]
n_folders = len(valid_directories)
n_images = 7

# Set up the plot
fig, axs = plt.subplots(n_folders, n_images, figsize=(n_folders*2, n_images * 2), constrained_layout=True)
# fig.suptitle('Main title')


# Traverse directories and plot images
for i, subdir in enumerate(valid_directories):
    if subdir in experiment_titles:  # Check if the directory is in the dictionary
        image_dir = os.path.join(main_dir, subdir, "fid_images/fake")
        images = select_images(image_dir, n_images)
        # Set the titles
        if i < n_folders:
            axs[i, n_images // 2].set_title(experiment_titles[subdir], fontsize=26, loc='center', pad=0)

        # Display each image in the appropriate subplot
        for j, img in enumerate(images):
            if i < n_folders and j < n_images:
                axs[i, j].imshow(img)
                axs[i, j].axis('off')  # Turn off axis

plt.tight_layout(pad=1, w_pad=0, h_pad=1)
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
