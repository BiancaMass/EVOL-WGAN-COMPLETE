import os
from datetime import datetime
import math


current_time = datetime.now()
STRING_TIME = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
OUTPUT_DIR = os.path.join(f"./output/{STRING_TIME}/")
EVOLUTIONARY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'evol')
GAN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'gan')

INPUT_DIR = os.path.join(f'./input')
DATASET_DIR = os.path.join(INPUT_DIR, 'datasets')

SEED = 97

# --------------------------------
#      INPUT DATA PARAMETERS
# --------------------------------
GAN_TRAIN_SIZE = 1000
GAN_VALID_SIZE = 250
EVOL_TRAIN_SIZE = 400
EVOL_VALID_SIZE = 100

# --------------------------------
#        IMAGE PARAMETERS
# --------------------------------
IMAGE_SIDE = 28
CLASSES = [0, 1]  # This only works for MNIST, picks number classes as specified in list
N_PIXELS = IMAGE_SIDE ** 2  # Assumes square images
N_CHANNELS = 1

# --------------------------------
#        CIRCUIT PARAMETERS
# --------------------------------
N_PATCHES = 28
PIXELS_PER_PATCH = int(N_PIXELS / N_PATCHES)
PATCH_WIDTH = int(IMAGE_SIDE)  # Each patch is assumed to have width = image_width
PATCH_HEIGHT = int((N_PIXELS / N_PATCHES) / PATCH_WIDTH)
# Data qubit = number of qubits required to generate as many pixels as needed PER PATCH
N_DATA_QUBITS = math.ceil(math.log(int((IMAGE_SIDE * IMAGE_SIDE) / N_PATCHES), 2))
N_ANCILLAS = 1

