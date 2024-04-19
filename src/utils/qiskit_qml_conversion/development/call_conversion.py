import numpy as np
from qiskit import QuantumCircuit
import pennylane as qml
import torch

from conversion_qiskit_pennylane import ConversionQiskitPenny

device = torch.device("cpu")


# Example Qiskit circuit to convert
n_qubits = 7
qc = QuantumCircuit(n_qubits)
for i in range(n_qubits):
    qc.ry(0.4, i)
qc.rx(0.9, 0)  # angle, qubit index
qc.h(1)  # qubit index
qc.ry(0.5, 1)  # angle, qubit index
qc.rz(0.6, 2)  # angle, qubit index
qc.h(2)  # qubit index
qc.h(2)  # qubit index
qc.h(5)  # qubit index
qc.rxx(0.7, 2, 3)  # angle, qubit1, qubit2
qc.ryy(0.2, 4, 5)  # angle, qubit1, qubit2
qc.rzz(0.2, 5, 6)  # angle, qubit1, qubit2


# Assuming you have the path to your QASM file
# qasm_file_path = "/Users/bmassacci/Desktop/results/Evolutionary/28-patches/u_cnot/24_03_26_19_16_14/final_best_circuit.qasm"

# Read the QASM file
# with open(qasm_file_path, 'r') as file:
#     qasm_str = file.read()

# Create a QuantumCircuit from QASM string
# qc = QuantumCircuit.from_qasm_str(qasm_str)

print(f'Quantum circuit before removing the encoding layer: {qc}')

del qc.data[0:qc.num_qubits]

print(f'Quantum circuit after removing the encoding layer: {qc}')

# Example usage
# latent_vector = torch.rand(n_qubits, device=device)
latent_vector = np.random.uniform(low=0.0, high=1.0, size=qc.num_qubits)
generator = ConversionQiskitPenny(quantum_circuit=qc, latent_vector=latent_vector)
pennylane_circuit = generator.variational_block()
probs = generator.get_probability_vector()

# Draw the PennyLane circuit
drawn_circuit = qml.draw(pennylane_circuit)
print(drawn_circuit(latent_vector))
print(f'\nLength prob vector: {len(probs)},\nSum of prob vector: {sum(probs)}')



