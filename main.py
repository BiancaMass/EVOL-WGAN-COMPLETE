import torch
import os

from src.utils.image_utils.data_loader_MNIST import dataloader_mnist
from src.evolutionary import QES_GAN as qes_gan

import configs.general_configs as general_configs
import configs.config_escqwgan as es_configs
import configs.config_gan as gan_config


def main():
    # Common parameters
    evol_output_dir = general_configs.EVOLUTIONARY_OUTPUT_DIR
    gan_output_dir = general_configs.GAN_OUTPUT_DIR

    dataset_dir = general_configs.DATASET_DIR

    # Create evolutionary output directory
    if not os.path.exists(evol_output_dir):
        os.makedirs(evol_output_dir)
        print(f"Evolutionary output directory created: {evol_output_dir}")

    # Create GAN output directory
    if not os.path.exists(gan_output_dir):
        os.makedirs(gan_output_dir)
        print(f"GAN output directory created: {gan_output_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    num_workers = 1 if device == 'cpu' else 8 if device == 'cuda' else 0
    print(f"Number of workers selected: {num_workers}")

    # Parameters for both GAN and Evol
    classes = general_configs.CLASSES
    n_data_qubits = general_configs.N_DATA_QUBITS
    n_ancilla = general_configs.N_ANCILLAS
    image_side = general_configs.IMAGE_SIDE
    patch_shape = (general_configs.PATCH_WIDTH, general_configs.PATCH_HEIGHT)
    n_pixels_patch = general_configs.PIXELS_PER_PATCH
    n_patches = general_configs.N_PATCHES
    randn = gan_config.RANDN
    n_channels = general_configs.N_CHANNELS

    # Params just for evol
    evol_batch_size = es_configs.EVOL_BATCH_SIZE
    n_children = es_configs.N_CHILDREN
    n_max_evaluations = es_configs.M_MAX_EVALUATIONS
    shots = es_configs.SHOTS
    dtheta = es_configs.DTHETA
    action_weights = es_configs.ACTION_WEIGHTS
    multi_action_pb = es_configs.MULTI_ACTION_PB
    max_gen_no_improvements = es_configs.MAX_GEN_NO_IMPROVEMENT
    max_depth = es_configs.MAX_DEPTH

    evol_train_size = general_configs.EVOL_TRAIN_SIZE
    evol_val_size = general_configs.EVOL_VALID_SIZE

    # Params just for gan
    gan_batch_size = gan_config.GAN_BATCH_SIZE
    gan_n_epochs = gan_config.N_EPOCHS
    gan_n_layers = gan_config.N_LAYERS

    gan_train_size = general_configs.GAN_TRAIN_SIZE
    gan_val_size = general_configs.GAN_VALID_SIZE

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'\nUsing {device} as a device\n')

    # Load the training and validation images for both algorithms
    train_gan_loader, val_gan_loader, train_evo_loader, val_evo_loader = dataloader_mnist(
        num_workers=num_workers,
        data_dir=dataset_dir,
        image_side=image_side,
        classes=classes,
        train=True,
        evol_batch_size=evol_batch_size,
        gan_batch_size=gan_batch_size,
        gan_train_size=gan_train_size,
        gan_val_size=gan_val_size,
        evol_train_size=evol_train_size,
        evol_val_size=evol_val_size
    )

    # Initialize a dictionary to hold arguments for Qes
    qes_args = {
        'n_data_qubits': n_data_qubits,
        'n_ancilla': n_ancilla,
        'patch_shape': patch_shape,
        'pixels_per_patch': n_pixels_patch,
        'batch_size': evol_batch_size,
        'classes': classes,
        'n_children': n_children,
        'n_patches': n_patches,
        'n_max_evaluations': n_max_evaluations,
        'shots': shots,
        'device': device,
        'dtheta': dtheta,
        'action_weights': action_weights,
        'multi_action_pb': multi_action_pb,
        'max_gen_no_improvement': max_gen_no_improvements,
        'max_depth': max_depth,
        'output_dir': evol_output_dir
    }

    print("*** STARTING EVOLUTIONARY SEARCH ***")
    qes = qes_gan.Qes(**qes_args)
    qes.data()

    # Save the output
    # Use the input to call and train the GAN


if __name__ == '__main__':
    main()