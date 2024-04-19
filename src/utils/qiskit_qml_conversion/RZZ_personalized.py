import pennylane as qml
from pennylane.operation import Operation
import numpy as np


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


