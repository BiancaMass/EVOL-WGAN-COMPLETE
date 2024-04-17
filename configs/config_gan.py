from datetime import datetime
import os
import csv

CURRENT_TIME = datetime.now()
STRING_TIME = CURRENT_TIME.strftime("%Y-%m-%d-%H%M")
OUTPUT_DIR = os.path.join(f"./output/0_imported_circuits/{STRING_TIME}")

INPUT_FOLDER = "/Users/bmassacci/main_folder/maastricht/academics/quantum_thesis/scripts/B-WGAN" \
               "-Evol/input/24_03_26_19_16_14/"

QASM_FILE_PATH= INPUT_FOLDER + "final_best_circuit.qasm"
METADATA_FILE_PATH = INPUT_FOLDER + "metadata.csv"

#### Image parameters ####
CLASSES = [0,1]
IMAGE_SIZE = 28
N_CHANNELS = 1  # MNIST is grayscale

#### Training parameters ####
RANDN = False
BATCH_SIZE = 25
N_EPOCHS = 50
N_LAYERS = 2
