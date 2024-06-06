import os
import random
from PIL import Image

input_folder = '/Volumes/SANDISK32_2/final_round5/24_05_18_21_36_23/fid_images/fake'
output_folder = '/Users/bmassacci/main_folder/maastricht/academics/quantum_thesis/images/mega_comparative_plot'
file_name = 'F04_FID.png'
file_destination = os.path.join(output_folder, file_name)

# Get list of images in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')]

# Sample 25 images randomly
sampled_images = random.sample(image_files, 25)

# Create a blank image for the grid
final_image = Image.new('RGB', (28 * 5, 28 * 5))

# Paste images into the grid
for i, image_file in enumerate(sampled_images):
    img = Image.open(os.path.join(input_folder, image_file))
    img = img.resize((28, 28))  # Ensure the image is 28x28
    x = (i % 5) * 28
    y = (i // 5) * 28
    final_image.paste(img, (x, y))

# Save the final image
final_image.save(file_destination)

print(f"Saved the grid image to {file_destination}")
