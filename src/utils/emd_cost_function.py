import scipy
from scipy import stats
import torch

from src.evolutionary.nets.generator_methods import from_patches_to_image, from_probs_to_pixels


def emd_scoring_function(real_images_preloaded, batch_size, qc,
                         n_tot_qubits, n_ancillas, n_patches,
                         pixels_per_patch, patch_width, patch_height, sim):
    """
    Computes the Earth Mover's Distance (EMD), aka Wasserstein distance, between a batch of real
    images and a batch of images generated by a quantum circuit. The goal is to evaluate the
    quality of the images generated by the quantum circuit. A lower EMD score indicates that the
    distribution of the generated images is closer to the target distribution (real images).
    Hence lower score indicates better quantum circuit. A score of 0 indicates that the
    distributions are identical. Note: EMD has no upper bound.

    :param real_images_preloaded: tensor. Preloaded real images.
    :param batch_size: int. The number of images to generate and evaluate in one batch.
    :param qc: qiskit.circuit.quantumcircuit.QuantumCircuit. The qc that generates images.
    :param n_tot_qubits: int. The total number of qubits in the quantum circuit.
    :param n_ancillas: int. The number of ancillary qubits in the quantum circuit.
    :param n_patches: int. The number of patches that each image is divided into for processing.
    :param pixels_per_patch: int. The number of pixels that constitute each patch.
    :param patch_width: int. The width (in pixels) of each image patch.
    :param patch_height: int. The height (in pixels) of each image patch.
    :param sim: StatevectorSimulator. The simulator that executes the qc to generate image patches.

    :return: float. The Earth Mover's Distance between the real and generated images.
    """
    generated_images_list = []
    if n_patches > 1:
        for batch_index in range(batch_size):
            generated_image = from_probs_to_pixels(quantum_circuit=qc,
                                                   n_tot_qubits=n_tot_qubits,
                                                   n_ancillas=n_ancillas,
                                                   sim=sim)[:pixels_per_patch]

            generated_image = generated_image.reshape(1, patch_height, patch_width)
            generated_images_list.append(generated_image)

    else:
        for batch_index in range(batch_size):
            generated_image = from_patches_to_image(quantum_circuit=qc,
                                                    n_tot_qubits=n_tot_qubits,
                                                    n_ancillas=n_ancillas,
                                                    n_patches=n_patches,
                                                    pixels_per_patch=pixels_per_patch,
                                                    patch_width=patch_width,
                                                    patch_height=patch_height,
                                                    sim=sim)
            generated_images_list.append(generated_image)

    real_images_tensor = real_images_preloaded
    generated_images_tensor = torch.stack([torch.from_numpy(image).float() for image in generated_images_list])

    real_images_flat = real_images_tensor.view(real_images_tensor.size(0), -1)
    generated_images_flat = generated_images_tensor.view(generated_images_tensor.size(0), -1)

    generated_images_flat_np = generated_images_flat.cpu().detach().numpy()
    real_images_flat_np = real_images_flat.cpu().detach().numpy()

    distance_real_gen = scipy.stats.wasserstein_distance(real_images_flat_np.flatten(),
                                                         generated_images_flat_np.flatten())

    return distance_real_gen
