from qiskit import QuantumCircuit, QuantumRegister


def state_embedding(circuit, n_tot_qubits, latent_vector):
    """Takes any Qiskit quantum circuit and adds the state embedding layer BEFORE all other
    gates. The state embedding is composed of RY rotational gates with parameters elements
    from a latent vector drawn from a certain latent space (eg., uniform between [0,1])

    :param circuit: qiskit.circuit.QuantumCircuit. The quantum circuit to which apply state
                    embedding.
    :param n_tot_qubits: int. Number of total qubits of the circuit.
    :param latent_vector: np.array: Array of angles for the RY gates, of length = n_qubits of the circuit

    """
    embedding_layer = QuantumCircuit(QuantumRegister(n_tot_qubits, 'qubit'))
    # State embedding: Ry rotation using the latent vector
    for i in range(n_tot_qubits):
        embedding_layer.ry(latent_vector[i], i)

    complete_circuit = embedding_layer.compose(circuit, front=True)

    return complete_circuit