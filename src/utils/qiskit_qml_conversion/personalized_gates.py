"""
Customized RXX, RYY, RZZ gates for Pennylane. Source for how to build your own customized gate:
https://pennylane.ai/blog/2021/05/how-to-add-custom-gates-and-templates-to-pennylane/
"""


import pennylane as qml
from pennylane.operation import Operation
import numpy as np


# Define the custom RXX gate inheriting from Pennylane Operation class
class RXX(Operation):
    num_params = 1  # This operation takes one parameter (e.g., rotation angle theta)
    num_wires = 2  # The operation acts on two quantum wires (qubits).
    par_domain = "R"  # the domain of the parameter; "R" means the parameter is a real number.

    grad_method = "A"  # Specifies that the gradient is analytic (parameter shift rule)
    grad_recipe = None  # Uses the default gradient recipe. Optional but done for clarity.

    # Defines the generator of the operation, which is used in calculating gradients.
    # For some optimizers the generating operator for the gate is used. For our gate, we provide
    # PennyLane with this information via this line, where the first entry is is the operator (the
    # matrix representation of X⊗X (the Kronecker
    # product of Pauli X with itself)) and the second entry is a scaling prefactor -0.5.
    generator = [(qml.PauliX(0) @ qml.PauliX(1)).matrix, -0.5]

    # A static method that defines how this operation can be decomposed into other PennyLane
    # operations. It decomposes the RXX operation into a PauliRot operation with the given
    # parameters and wires
    @staticmethod
    def compute_decomposition(theta, wires):
        return [qml.PauliRot(theta, 'XX', wires=wires)]

    # A property that computes the matrix representation of the RXX operation given its parameter
    # (theta). Basically a 2-qubits rotation matrix.
    @property
    def _matrix(*params):
        theta = params[0]
        c = np.cos(0.5 * theta)
        s = np.sin(0.5 * theta)
        return np.array(
            [
                [c, 0, 0, -s],
                [0, c, -s, 0],
                [0, -s, c, 0],
                [-s, 0, 0, c]
            ]
        )

    # Defines the adjoint (inverse) of the operation. In this case is the RXX operation
    # with the negative of its original parameter.
    def adjoint(self):
        return RXX(-self.data[0], wires=self.wires)


# Define the custom RYY gate inheriting from Pennylane Operation class
class RYY(Operation):
    num_params = 1  # This operation takes one parameter (e.g., rotation angle theta)
    num_wires = 2  # The operation acts on two quantum wires (qubits).
    par_domain = "R"  # The domain of the parameter; "R" means the parameter is a real number.

    grad_method = "A"  # Specifies that the gradient is analytic (parameter shift rule)
    grad_recipe = None  # Uses the default gradient recipe. Optional but done for clarity.

    # Defines the generator of the operation, which is used in calculating gradients.
    # It's a tuple where the first element is the matrix representation of Y⊗Y (the Kronecker
    # product of Pauli Y with itself) and the second element is a scaling factor -0.5.
    generator = [(qml.PauliY(0) @ qml.PauliY(1)).matrix, -0.5]

    # A static method that defines how this operation can be decomposed into other PennyLane
    # operations. It decomposes the RYY operation into a PauliRot operation with the given
    # parameters and wires.
    @staticmethod
    def compute_decomposition(theta, wires):
        return [qml.PauliRot(theta, 'YY', wires=wires)]

    # A property that computes the matrix representation of the RYY operation given its parameter
    # (theta). Basically a 2-qubits rotation matrix.
    @property
    def _matrix(*params):
        theta = params[0]
        c = np.cos(0.5 * theta)
        s = np.sin(0.5 * theta)
        return np.array(
            [
                [c, 0, 0, s],
                [0, c, -s, 0],
                [0, -s, c, 0],
                [s, 0, 0, c]
            ]
        )

    # Defines the adjoint (inverse) of the operation. In this case, it is the RYY operation
    # with the negative of its original parameter.
    def adjoint(self):
        return RYY(-self.data[0], wires=self.wires)


# Define the custom RZZ gate inheriting from Pennylane Operation class
class RZZ(Operation):
    num_params = 1  # This operation takes one parameter (e.g., rotation angle theta)
    num_wires = 2  # The operation acts on two quantum wires (qubits).
    par_domain = "R"  # The domain of the parameter; "R" means the parameter is a real number.

    grad_method = "A"  # Specifies that the gradient is analytic (parameter shift rule)
    grad_recipe = None  # Uses the default gradient recipe. Optional but done for clarity.

    # Defines the generator of the operation, which is used in calculating gradients.
    # It's a tuple where the first element is the matrix representation of Z⊗Z (the Kronecker
    # product of Pauli Z with itself) and the second element is a scaling factor -0.5.
    generator = [(qml.PauliZ(0) @ qml.PauliZ(1)).matrix, -0.5]

    # We add the method decomposition to relate our new gate to the gates that PennyLane already
    # knows.
    @staticmethod
    def compute_decomposition(theta, wires):
        return [qml.PauliRot(theta, 'ZZ', wires=wires)]

    # A property that computes the matrix representation of the RZZ operation given its parameter
    # (theta). This represents a 2-qubits rotation around the ZZ axis.
    @property
    def _matrix(*params):
        theta = params[0]
        return np.array(
            [
                [np.exp(-1j * 0.5 * theta), 0, 0, 0],
                [0, np.exp(1j * 0.5 * theta), 0, 0],
                [0, 0, np.exp(1j * 0.5 * theta), 0],
                [0, 0, 0, np.exp(-1j * 0.5 * theta)]
            ]
        )

    # Defines the adjoint (inverse) of the operation. In this case, it is the RZZ operation
    # with the negative of its original parameter.
    def adjoint(self):
        return RZZ(-self.data[0], wires=self.wires)


