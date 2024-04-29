import torch
import os

import configs.general_configs as general_configs
import configs.config_evol as es_configs
import configs.config_gan as gan_config
from src.utils.image_utils.data_loader_MNIST import dataloader_mnist
from src.evolutionary import QES_Gen as qes_g
from src.gan import CQWGAN as gan
from src.utils.plot_utils.gan_output_plots import plot_gan_outputs
from src.utils.set_seeds import set_seeds
from src.utils.gan_utils.fid_score_calc import fid_score_calculator
from src.utils.gan_utils.find_latest_saved_generator import find_latest_generator_file


def main():
    # set_seeds(general_configs.SEED)
    # ------------- Create directories for storing OUTPUT -------------
    evol_output_dir = general_configs.EVOLUTIONARY_OUTPUT_DIR
    gan_output_dir = general_configs.GAN_OUTPUT_DIR
    dataset_dir = general_configs.DATASET_DIR

    if not os.path.exists(evol_output_dir):
        os.makedirs(evol_output_dir)
        print(f"Evolutionary output directory created: {evol_output_dir}")

    if not os.path.exists(gan_output_dir):
        os.makedirs(gan_output_dir)
        print(f"GAN output directory created: {gan_output_dir}")
    # Note: GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    num_workers = 0 if device == 'cpu' else 8 if device == 'cuda' else 0
    print(f"Number of workers selected: {num_workers}")

    # ------------------- Initiate Parameters -------------------
    # Parameters for both GAN and EVOL
    classes = general_configs.CLASSES
    n_data_qubits = general_configs.N_DATA_QUBITS
    n_ancilla = general_configs.N_ANCILLAS
    image_side = general_configs.IMAGE_SIDE
    patch_shape = (general_configs.PATCH_HEIGHT, general_configs.PATCH_WIDTH)
    n_pixels_patch = general_configs.PIXELS_PER_PATCH
    n_patches = general_configs.N_PATCHES
    randn_latent = gan_config.RANDN
    n_channels = general_configs.N_CHANNELS

    # Parameters just for EVOL
    evol_batch_size = es_configs.EVOL_BATCH_SIZE
    evol_n_batches = es_configs.EVOL_N_BATCHES
    evol_batch_subset = es_configs.BATCH_SUBSET
    n_children = es_configs.N_CHILDREN
    n_max_evaluations = es_configs.M_MAX_EVALUATIONS
    dtheta = es_configs.DTHETA
    patch_for_evaluation = es_configs.PATCH_FOR_EVALUATION
    action_weights = es_configs.ACTION_WEIGHTS
    multi_action_pb = es_configs.MULTI_ACTION_PB
    max_gen_until_change = es_configs.MAX_GEN_UNTIL_CHANGE
    max_gen_no_improvement = es_configs.MAX_GEN_NO_IMPROVEMENT
    max_depth = es_configs.MAX_DEPTH

    gen_saving_frequency = es_configs.GENERATION_SAVING_FREQUENCY

    evol_train_size = general_configs.EVOL_TRAIN_SIZE
    evol_val_size = general_configs.EVOL_VALID_SIZE

    # Parameters just for GAN
    gan_batch_size = gan_config.GAN_BATCH_SIZE
    gan_n_epochs = gan_config.N_EPOCHS
    gan_n_layers = gan_config.N_LAYERS

    gan_train_size = general_configs.GAN_TRAIN_SIZE
    gan_val_size = general_configs.GAN_VALID_SIZE

    general_metadata_path = os.path.join(general_configs.OUTPUT_DIR, 'metadata.txt')

    # Creating a dictionary of all variables to save
    metadata_dict = {
        'Device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'Num Workers': 0 if device == 'cpu' else 8,
        'Classes': classes,
        'Data Qubits': n_data_qubits,
        'Ancillas': n_ancilla,
        'Image Side': image_side,
        'Patch Shape': patch_shape,
        'Pixels per Patch': n_pixels_patch,
        'Number of Patches': n_patches,
        'Randn Latent': randn_latent,
        'Channels': n_channels,
        'EVOL Batch Size': evol_batch_size,
        'EVOL Number of Batches': evol_n_batches,
        'EVOL Batch Subset': evol_batch_subset,
        'Number of Children': n_children,
        'Max Evaluations': n_max_evaluations,
        'DTheta': dtheta,
        'Patch for Evaluation': patch_for_evaluation,
        'Action Weights': action_weights,
        'Multi-action Probability': multi_action_pb,
        'Max Generation Until Change': max_gen_until_change,
        'Max Generation No Improvement': max_gen_no_improvement,
        'Max Depth': max_depth,
        'Generation Saving Frequency': gen_saving_frequency,
        'EVOL Training Size': evol_train_size,
        'EVOL Validation Size': evol_val_size,
        'GAN Batch Size': gan_batch_size,
        'GAN Epochs': gan_n_epochs,
        'GAN Layers': gan_n_layers,
        'GAN Training Size': gan_train_size,
        'GAN Validation Size': gan_val_size
    }

    # Writing metadata to file
    with open(general_metadata_path, 'w') as f:
        for key, value in metadata_dict.items():
            f.write(f'{key}: {value}\n')

    print("Creating the data loaders with the following dimensions:")
    print(f'EVOL batch size: {evol_batch_size} \n'
          f'GAN batch size: {gan_batch_size} \n'
          f'EVOL [training size, validation size]: [{evol_train_size},{evol_val_size}] \n'
          f'GAN [training size, validation size]: [{gan_train_size},{gan_val_size}]')
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
        'n_patches': n_patches,
        'dataloader': train_evo_loader,
        'evol_batch_size': evol_batch_size,
        'n_batches': evol_n_batches,
        'batch_subset': evol_batch_subset,
        'classes': classes,
        'n_children': n_children,
        'n_max_evaluations': n_max_evaluations,
        'dtheta': dtheta,
        'action_weights': action_weights,
        'multi_action_pb': multi_action_pb,
        'patch_for_evaluation': patch_for_evaluation,
        'device': device,
        'max_gen_until_change': max_gen_until_change,
        'max_gen_no_improvement': max_gen_no_improvement,
        'gen_saving_frequency': gen_saving_frequency,
        'output_dir': evol_output_dir,
        'max_depth': max_depth
    }

    print("\n*** INITIALIZING EVOLUTIONARY ALGORITHM ***\n")
    qes = qes_g.Qes(**qes_args)
    # Save the output
    print("\n*** STARTING EVOLUTIONARY SEARCH ***\n")
    qes.data()
    print("\n*** EVOLUTIONARY OUTPUT SAVED ***\n")
    # TODO: add saving of sample images generated with the architecture

    print("\n*** STARTING GAN TRAINING ***\n")

    qasm_file_path = os.path.join(evol_output_dir, 'final_best_circuit.qasm')
    metadata_file_path = os.path.join(evol_output_dir, 'metadata.csv')

    try:
        gan.train_imported_gan(train_dataloader=train_gan_loader,
                               classes=classes,
                               out_folder=gan_output_dir,
                               qasm_file_path=qasm_file_path,
                               metadata_file_path=metadata_file_path,
                               normal_latent=randn_latent,
                               image_side=image_side,
                               n_channels=n_channels,
                               n_layers=gan_n_layers,
                               batch_size=gan_batch_size,
                               n_epochs=gan_n_epochs)

    finally:
        plot_gan_outputs(input_csv_file=os.path.join(gan_output_dir, "training_values_history.csv"),
                         output_dir=gan_output_dir)

        # Calculate and save FID score for last generator
        # gen_file_path = find_latest_generator_file(gan_output_dir)
        # fid_score = fid_score_calculator(validation_dataloader=val_gan_loader,
        #                                  n_images_to_evaluate=250,
        #                                  path_to_last_generator=gen_file_path,
        #                                  qasm_file_path=qasm_file_path)
        # fid_score_file = os.path.join(gan_output_dir, 'fid_score.csv')
        # with open(fid_score_file, 'a') as file:
        #     file.write(f"{fid_score},{gen_file_path}\n")


if __name__ == '__main__':
    main()
