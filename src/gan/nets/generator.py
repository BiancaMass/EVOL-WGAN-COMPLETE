import pennylane as qml
import torch
import torch.nn as nn
from qiskit import QuantumCircuit

from src.utils.qiskit_qml_conversion.personalized_gates import RXX, RYY, RZZ


class QuantumGeneratorImported(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, image_shape, qasm_file_path, n_ancillas, n_sub_generators, n_layers):
        """Initialize the QuantumGenerator class.

        Args:
            image_shape (tuple): The shape of the output image. A tuple (channels, height, width).
            qasm_file_path (str): The path to the QASM file that contains the quantum circuit.
            n_ancillas (int): The number of ancillary qubits included in the quantum circuit.
            n_sub_generators (int): The number of sub generators (one per patch).
            n_layers (int): The number of times to repeat the imported ansatz for (layers).
        Returns:
            None
        """
        super().__init__()
        self.qasm_file_path = qasm_file_path
        self.image_shape = image_shape

        # Import and convert the circuit
        # It already deletes the encoding layer from the imported circuit
        self.qiskit_circuit, self.n_qubits, initial_params = self.importing_circuit()
        self.q_device = qml.device("default.qubit", wires=self.n_qubits)
        self.n_ancillas = n_ancillas
        self.n_sub_generators = n_sub_generators
        self.num_layers = n_layers

        self.tot_image_pixes = image_shape[2] ** 2
        self.pixels_per_patch = self.tot_image_pixes // n_sub_generators

        # This will have as many elements as number of sub-gens, and in each is a flat vector
        # with as many elements as num_layers * (sum(num_gate_per_layer *
        # num_params_for_that_gate))
        self.params = nn.ParameterList()  # Holds parameters for each sub-generator

        for subgen_idx in range(n_sub_generators):
            # Accumulate all parameters for a single sub-generator into a single tensor
            subgen_params_tensor = torch.tensor([])  # Start with an empty tensor

            for layer_idx in range(n_layers):
                layer_params = []
                for param in initial_params:
                    if isinstance(param, list):  # For gates requiring multiple parameters
                        # It actually performs better with random weight initialization
                        # layer_params.append(torch.tensor(param, dtype=torch.float32))
                        layer_params.append(torch.rand((len(param),), dtype=torch.float32))
                    else:  # For gates requiring a single parameter
                        # layer_params.append(torch.tensor(param[0], dtype=torch.float32))
                        layer_params.append(torch.rand((1,), dtype=torch.float32))

                # Convert layer parameters to a tensor and concatenate with the sub-generator tensor
                layer_params_tensor = torch.cat(layer_params, dim=0)
                subgen_params_tensor = torch.cat((subgen_params_tensor, layer_params_tensor),
                                                 dim=0)

            # Add the complete set of parameters for the sub-generator as an nn.Parameter
            self.params.append(nn.Parameter(subgen_params_tensor, requires_grad=True))

        self.qnode = qml.QNode(func=self.pennylane_circuit,  # defined below
                               device=self.q_device,  # the pennylane device initialized above
                               interface="torch")  # The interface for classical backpropagation

    def importing_circuit(self):
        """
        Imports a circuit from a QASM file and preprocesses it for use with Pennylane.

        Returns:
            QuantumCircuit: The imported Qiskit quantum circuit.
            int: The number of qubits in the circuit.
            list: A list of initial parameters for the quantum gates present in the circuit.
        """
        with open(self.qasm_file_path, 'r') as file:
            qasm_str = file.read()
        qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)
        # Delete the encoding layer
        del qiskit_circuit.data[0:qiskit_circuit.num_qubits]

        n_qubits = qiskit_circuit.num_qubits

        initial_params = []
        for instr, _, _ in qiskit_circuit.data:
            if instr.name.lower() in ["rx", "ry", "rz", "rxx", "ryy", "rzz"]:
                initial_params.append(instr.params[0])
            elif instr.name.lower() in ["u"]:
                initial_params.append(instr.params)

        return qiskit_circuit, n_qubits, initial_params

    def pennylane_circuit(self, latent_vector, params):
        """
        Converts and executes Qiskit circuit in PennyLane. Maps Qiskit gates to PennyLane
        equivalents, applies them with given parameters, and returns the probabilities of the
        computational basis states, of length 2^n_tot_qubits.

        Args:
            latent_vector (torch.Tensor): Input latent vector.
            params (list): Parameters for quantum gates.

        Returns:
            torch.Tensor: Probabilities of quantum states (computational basis states vector).
        """
        # Encode the latent vector
        for i, angle in enumerate(latent_vector):
            qml.RY(angle, wires=i)

        # Map Qiskit gates to PennyLane gates
        for layer_index in range(self.num_layers):  # repeat as many times as num_layers
            param_idx = layer_index  # Initialize parameter index
            for instr, qubits, _ in self.qiskit_circuit.data:  # Instructions, qubits, empty
                name = instr.name.lower()  # gate names all lower case
                wires = [q._index for q in qubits]  # wires for each single and double gate

                if name in ["rx", "ry", "rz"]:
                    # print(f'rx,ry,rz. Found gate {name} with param {instr.params} on qubit {wires}')
                    getattr(qml, name.upper())(params[param_idx], wires=wires)
                    param_idx += 1
                elif name == "rxx":
                    # print(f'RXX. Found gate {name} with param {instr.params} on qubit '
                    #       f'{wires[0]},{wires[1]}')
                    RXX(params[param_idx], wires=[wires[0], wires[1]])
                    param_idx += 1
                elif name == "ryy":
                    # print(f'RYY. Found gate {name} with param {instr.params} on qubit '
                    #       f'{wires[0]},{wires[1]}')
                    RYY(params[param_idx], wires=[wires[0], wires[1]])
                    param_idx += 1
                elif name == "rzz":
                    # print(f'RZZ. Found gate {name} with param {instr.params} on qubit '
                    #       f'{wires[0]},{wires[1]}')
                    RZZ(params[param_idx], wires=[wires[0], wires[1]])
                    param_idx += 1
                elif name == "u":
                    # Gotta do this way cause the list of params for each (sub)-gen is flat now
                    qml.Rot(params[param_idx],
                            params[param_idx+1],
                            params[param_idx+2],
                            wires=wires)
                    param_idx += 3  # because you are using three params from a flat param list
                elif name == "cx":
                    qml.CNOT(wires=[wires[0], wires[1]])
                elif name == "h":
                    # print(f'h. Found gate {name} on qubit {wires}')
                    qml.Hadamard(wires=wires[0])  # hadamard has no parameters

        return qml.probs(wires=range(self.n_qubits))

    def forward(self, x):
        """
        Perform a forward pass through the QuantumGenerator. Generates one image (tensor)
        given a latent vector.
        :param x: torch.Tensor. Input tensor (latent vector).

        :returns: torch.Tensor. Output tensor (image or image patch).
        """
        # Create a tensor to store the output images
        output_images = torch.Tensor(x.size(0), 0).to(x.device)

        for subgen_params in self.params:  # Loop among each sub_generator
            patches = torch.Tensor(0, self.pixels_per_patch).to(x.device)  # to store the patches
            # Generate patches for each item in x, the latent vector (input tensor)
            for latent_vec in x:  # x has shape (batch_size, n_tot_qubits)
                patch = self.generate_patch(latent_vec=latent_vec,
                                            subgen_params=subgen_params,
                                            patch_size=self.pixels_per_patch)
                patches = torch.cat((patches, patch))
            output_images = torch.cat((output_images, patches), 1)

        final_out = output_images.view(output_images.shape[0], *self.image_shape).to(x.device)

        return final_out

    def generate_patch(self, latent_vec, subgen_params, patch_size):
        """
        Generate a patch using a sub-generator and a latent vector.

        :param latent_vector: torch.Tensor. Latent vector input.
        :param sub_gen_params: dict. Parameters for the gates in the sub-generator.
        :param patch_size: int. Size of the patch (to subset and keep only the needed qubits).

        :returns: torch.Tensor. Generated patch.
        """
        patch = self.partial_measure(latent_vec, subgen_params).float().unsqueeze(0)
        return patch[:, :patch_size]

    def partial_measure(self, latent_vector, weights):
        """
        Performs partial trace and post-processing operations on the quantum circuit outputs.

        :param latent_vector: torch.Tensor. Latent vector, the input to the circuit.
        :param weights: torch.Tensor. Tensor containing the weights of the quantum circuit.

        :return: torch.Tensor. Post-processed patch obtained from the circuit outputs.

        Sources that explain the procedure:
        https://discuss.pennylane.ai/t/ancillary-subsystem-measurement-then-trace-out/1532
        https://pennylane.ai/qml/demos/tutorial_quantum_gans/
        """
        probs = self.qnode(latent_vector, weights)
        # Introduce non-linearity
        probsgiven0 = probs[:2**(self.n_qubits - self.n_ancillas)]
        # Normalize the probabilities by their sum (normalization constraint)
        probsgiven0 /= torch.sum(probsgiven0)

        # Normalise pixels in [-1, 1] for the GAN (because we import MNIST with this range)
        post_processed_patch = ((probsgiven0 / torch.max(probsgiven0)) - 0.5) / 0.5

        return post_processed_patch  #.to(latent_vector.device)

