import numpy as np
import torch

from qiskit import execute


def get_probabilities(quantum_circuit, n_tot_qubits, sim):
    """ Executes the given circuit on the given simulator, calculates and outputs the
    probabilities of each computational quantum state for the circuit.
    The probabilities are calculated as the square of the absolute value of each amplitude
    in the statevector obtained from the simulation result.

    :param quantum_circuit: qiskit.circuit.quantumcircuit.QuantumCircuit. The quantum circuit to be executed.
    :param n_tot_qubits: int. The total number of qubits in the quantum circuit (data + ancilla).
    :param sim: str. Name of the simulator backend to be used for execution, e.g., 'aer_simulator'.

    :return: numpy.ndarray. An array of probabilities corresponding to each quantum state.
    """

    p = np.zeros(2 ** n_tot_qubits)  # to store the probabilities

    job = execute(quantum_circuit, sim)  # Execute the circuit `qc` on the simulator `sim`
    result = job.result()  # Retrieves the result of the execution
    statevector = result.get_statevector(quantum_circuit)
    # Calculate probabilities: (the square of the abs. value of amplitude of each element in s.v.)
    for i in range(len(np.asarray(statevector))):
        p[i] = np.absolute(statevector[i]) ** 2  # store probs in array `p`

    return p  # torch.from_numpy(p)


def from_probs_to_pixels(quantum_circuit, n_tot_qubits, n_ancillas, sim):
    """
    Converts quantum circuit probabilities to normalized pixel values.

    1. Calculates the probabilities of quantum states in a given circuit.
    2. Processes these probabilities to generate normalized pixel values, excluding ancilla qubit
    values to introduce non-linearity
    3. Ensures the sum of probabilities equals 1.
    4. Normalizes the final pixel values between -1 and 1 as required for the GAN images

    :param quantum_circuit: qiskit.circuit.quantumcircuit.QuantumCircuit. The quantum circuit to be executed.
    :param n_tot_qubits: int. Total number of qubits in the circuit.
    :param n_ancillas: int. Number of ancilla qubits in the circuit.
    :param sim: str. Name of the simulator backend to be used for execution, e.g., 'aer_simulator'.

    """
    # probs is a Tensor
    probs = get_probabilities(quantum_circuit=quantum_circuit, n_tot_qubits=n_tot_qubits,
                              sim=sim)
    # Introduce non-linearity
    probsgiven0 = probs[:2 ** (n_tot_qubits - n_ancillas)]
    # Normalize the probabilities by their sum (normalization constraint)
    probsgiven0 /= sum(probsgiven0)
    # Normalise pixels in [-1, 1] for the GAN (because we import MNIST with this range)
    post_processed_patch = ((probsgiven0 / max(probsgiven0)) - 0.5) / 0.5

    return post_processed_patch


def from_patches_to_image(quantum_circuit, n_tot_qubits, n_ancillas, n_patches, pixels_per_patch,
                          patch_width, patch_height, sim):
    """Constructs an image from quantum circuit generated patches. Iterates over a specified
    number of patches, generating each patch from a quantum circuit using the `from_probs_to_pixels`
    function. It then combines these patches to form a single image.

    :param quantum_circuit: qiskit.circuit.quantumcircuit.QuantumCircuit. The quantum circuit to be executed.
    :param n_tot_qubits: int. Total number of qubits in the circuit.
    :param n_ancillas: int. Number of ancilla qubits in the circuit.
    :param n_patches: int. Number of patches to generate for the image.
    :param pixels_per_patch: int. Number of pixels in each patch.
    :param patch_width: width (in pixels) of each patch.
    :param patch_height: height (in pixels) of each patch.
    :param sim: str. Name of the simulator backend to be used for execution, e.g., 'aer_simulator'.

    :return: torch.Tensor. A tensor representing the final image.
    """
    final_image = torch.empty((0, n_patches))
    for patch in range(n_patches):
        current_patch = from_probs_to_pixels(quantum_circuit=quantum_circuit,
                                             n_tot_qubits=n_tot_qubits,
                                             n_ancillas=n_ancillas,
                                             sim=sim)
        current_patch = current_patch[:pixels_per_patch]
        current_patch = torch.reshape(torch.from_numpy(current_patch),
                                      (1, patch_height, patch_width))
    if n_patches == 1:
        final_image = current_patch
    else:
        final_image = torch.cat((final_image, current_patch), dim=0)

    return final_image
