import torch
import numpy as np
import random
from qiskit import QuantumCircuit, QuantumRegister, Aer
from scipy.stats import wasserstein_distance_nd

from src.utils.image_utils.data_loader_MNIST import dataloader_mnist
from configs import general_configs, config_evol
from src.evolutionary.nets.generator_methods import from_probs_to_pixels, from_patches_to_image
from src.utils.evol_utils.evolutionary_image_utils import crop_images_for_evol
from src.utils.evol_utils.state_embedding import state_embedding, latent_creation

"""
This script calculates the EMD score for images produced with an *untrained* Parameterized
Quantum Circuit of three parameter rotation gates U(theta, phi, omega) followed by a layer of CNOTS.
"""


def calculate_benchmark_emd():
    ancilla = general_configs.N_ANCILLAS
    tot_qubits = general_configs.N_ANCILLAS + general_configs.N_DATA_QUBITS
    patch_for_cropping = config_evol.PATCH_FOR_EVALUATION
    patch_height, patch_width = general_configs.PATCH_HEIGHT, general_configs.PATCH_WIDTH
    pixels_per_patch = general_configs.PIXELS_PER_PATCH
    n_patches = general_configs.N_PATCHES
    latent_vector = np.random.rand(tot_qubits)
    n_batches_to_crop = config_evol.EVOL_N_BATCHES
    n_images_to_compare = config_evol.EVOL_BATCH_SIZE * config_evol.BATCH_SUBSET
    sim = Aer.get_backend('statevector_simulator')

    weights = [np.random.rand(3) for qubit in range(tot_qubits)]

    # Step 1: define the circuit with qiskit
    circ = QuantumCircuit(QuantumRegister(tot_qubits, 'qubit'))
    # Embedding done later so i can change the latent vector
    # for qubit in range(tot_qubits):
    #     circ.ry(latent_vector[qubit], qubit)
    for qubit in range(tot_qubits):
        circ.u(weights[qubit][0], weights[qubit][1], weights[qubit][2], qubit)
    for qubit in range(tot_qubits - 1):
        circ.cx(qubit, qubit + 1)

    # print(circ)

    # Step 2: load a few batches of the real dataset.
    dataloader, _, _, _ = dataloader_mnist(num_workers=0,
                                           data_dir=general_configs.DATASET_DIR,
                                           image_side=28,
                                           classes=[0, 1],
                                           train=True,
                                           evol_batch_size=100,
                                           gan_batch_size=25,
                                           gan_train_size=800,
                                           gan_val_size=200,
                                           evol_train_size=300,
                                           evol_val_size=100)

    cropped_real_images = crop_images_for_evol(dataloader=dataloader,
                                               patch_position=patch_for_cropping,
                                               patch_height=patch_height,
                                               n_batches=n_batches_to_crop,
                                               device='cpu')

    subset_cropped_real_images = random.sample(cropped_real_images, n_images_to_compare)

    # Step 2: generate images with a different latent vector for each, and untrained weights for fair
    # comparison

    generated_images_list = []
    if n_patches > 1:
        for batch_index in range(n_images_to_compare):
            qc_with_embedding = state_embedding(circ, tot_qubits, latent_creation(tot_qubits))
            generated_image = from_probs_to_pixels(quantum_circuit=qc_with_embedding,
                                                   n_tot_qubits=tot_qubits,
                                                   n_ancillas=ancilla,
                                                   sim=sim)[:pixels_per_patch]

            generated_image = generated_image.reshape(1, patch_height, patch_width)
            generated_images_list.append(generated_image)

    else:
        for batch_index in range(n_images_to_compare):
            qc_with_embedding = state_embedding(circ, tot_qubits, latent_creation(tot_qubits))
            generated_image = from_patches_to_image(quantum_circuit=qc_with_embedding,
                                                    n_tot_qubits=tot_qubits,
                                                    n_ancillas=ancilla,
                                                    n_patches=n_patches,
                                                    pixels_per_patch=pixels_per_patch,
                                                    patch_width=patch_width,
                                                    patch_height=patch_height,
                                                    sim=sim)
            generated_images_list.append(generated_image)

    real_images_tensor = torch.stack(subset_cropped_real_images, dim=0)
    generated_images_tensor = torch.stack(
        [torch.from_numpy(image).float() for image in generated_images_list])

    real_images_flat = real_images_tensor.reshape(n_images_to_compare, -1)
    generated_images_flat = generated_images_tensor.reshape(n_images_to_compare, -1)

    if real_images_flat.shape != generated_images_flat.shape:
        print("Warning: The input tensors used for EMD calculations do not have the same shape.")

    emd_real_gen = wasserstein_distance_nd(u_values=real_images_flat,
                                           v_values=generated_images_flat)

    return emd_real_gen


emd = calculate_benchmark_emd()
print(emd)
