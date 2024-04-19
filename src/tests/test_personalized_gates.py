import numpy as np
import pennylane as qml
from pennylane.operation import Operation

from src.utils.qiskit_qml_conversion.personalized_gates import RXX, RYY, RZZ


n_qubits = 2
dev = qml.device('default.qubit', wires=n_qubits, shots=1000)


def test_RXX_gate(theta):
    @qml.qnode(dev)
    def circuit():
        RXX(theta, wires=[0, 1])
        return qml.probs(wires=list(range(n_qubits)))

    result = circuit()
    print(f"RXX({theta}) probabilities:", result)

def test_RYY_gate(theta):
    @qml.qnode(dev)
    def circuit():
        RYY(theta, wires=[0, 1])
        return qml.probs(wires=list(range(n_qubits)))

    result = circuit()
    print(f"RYY({theta}) probabilities:", result)

def test_RZZ_gate(theta):
    @qml.qnode(dev)
    def circuit():
        RZZ(theta, wires=[0, 1])
        return qml.probs(wires=list(range(n_qubits)))

    result = circuit()
    print(f"RZZ({theta}) probabilities:", result)

theta_values = np.linspace(0, 2 * np.pi, 10)  # Testing across a range of theta values

for theta in theta_values:
    test_RXX_gate(theta)
    test_RYY_gate(theta)
    test_RZZ_gate(theta)