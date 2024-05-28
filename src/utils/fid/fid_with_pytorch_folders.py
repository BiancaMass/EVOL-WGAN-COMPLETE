import torch
import os
import math
from torchvision.utils import save_image
import gc

from src.gan.nets.generator import QuantumGeneratorImported
import configs.general_configs as general_configs
import configs.config_gan as gan_config

"""
Generates images given a circuit architecture file and a generator .pt file,
Stores them in a folder, so that they are ready to be used with pytorch_fid
for FID calculation
"""

device = 'cpu'
num_workers = 0 if device == 'cpu' else 8 if device == 'cuda' else 0
print(f"Number of workers selected: {num_workers}")
image_side = general_configs.IMAGE_SIDE
channels = general_configs.N_CHANNELS
classes = general_configs.CLASSES
randn_true = gan_config.RANDN

# STUFF TO CHANGE
patch_shape = (1, 28)
N_ANCILLAS = 1
current_folder = "24_05_18_16_40_18"

source_folder = "/Volumes/SANDISK32_2/final_round5"
last_generator = "generator-3180.pt"
# END STUFF TO CHANGE

n_sub_generators = int(image_side / (int(patch_shape[0])))
N_DATA_QUBITS = math.ceil(math.log(int((image_side * image_side) / n_sub_generators), 2))
n_layers = 1
n_tot_qubits = N_DATA_QUBITS + N_ANCILLAS


folder = os.path.join(source_folder, current_folder)

print(f'Going over folder {folder}')

output_folder = os.path.join(folder, "fid_images")

fake_folder = os.path.join(output_folder, "fake")
if not os.path.exists(fake_folder):
    os.makedirs(fake_folder)

path_to_qasm_file = os.path.join(folder, "evol", "final_best_circuit.qasm")
path_to_last_generator = os.path.join(folder, "gan", last_generator)

n_images_to_compare = 200

print("Initializing generator")
generator = QuantumGeneratorImported(image_shape=(channels, image_side, image_side),
                                     qasm_file_path=path_to_qasm_file,
                                     n_ancillas=general_configs.N_ANCILLAS,
                                     n_sub_generators=n_sub_generators,
                                     n_layers=n_layers)
generator = generator.to(device)
generator.load_state_dict(torch.load(path_to_last_generator, map_location=torch.device('cpu')))

# Generate fake images
print("Generating images")
z = torch.randn(n_images_to_compare, n_tot_qubits, device=device) if randn_true else \
    torch.rand(n_images_to_compare, n_tot_qubits, device=device)
fake_images = generator(z)

print("Saving the images")
for index, img in enumerate(fake_images):
    # Constructs file path
    file_path = os.path.join(fake_folder, f'image_{index + 1:03d}.png')
    # Save the image
    save_image(img, file_path)

print("Clearing cache")
del generator
gc.collect()
print("*** DONE ***")

# Then, on the terminal call (second pooling layer)
# python -m pytorch_fid path/to/dataset1 path/to/dataset2 --dims 192
