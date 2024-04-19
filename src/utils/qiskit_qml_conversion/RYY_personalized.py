import pennylane as qml
from pennylane.operation import Operation
import numpy as np


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


