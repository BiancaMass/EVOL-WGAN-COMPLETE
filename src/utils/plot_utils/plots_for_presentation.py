import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
import random
import os


# Define the function to generate and save the quantum circuits
def generate_quantum_circuits(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    num_qubits = 6
    circuits = []

    # Step 0: Circuit with only Ry gates
    qc0 = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        theta = random.uniform(0, 2 * 3.14159)  # Random angle between 0 and 2*pi
        qc0.ry(theta, qubit)
    circuits.append(qc0)

    # Step 1: Add a CNOT in a random position
    qc1 = qc0.copy()
    control = random.randint(0, num_qubits - 2)
    target = random.randint(control + 1, num_qubits - 1)
    qc1.cx(control, target)
    circuits.append(qc1)

    # Step 2: Add a U(three parameters) with random params in a random position
    qc2 = qc0.copy()
    params = [random.uniform(0, 2 * 3.14159) for _ in range(3)]
    position = random.randint(0, num_qubits - 1)
    qc2.u(params[0], params[1], params[2], position)
    circuits.append(qc2)

    # Step 3: Add another U(three parameters) with random params in a random position
    qc3 = qc0.copy()
    params = [random.uniform(0, 2 * 3.14159) for _ in range(3)]
    position = random.randint(0, num_qubits - 1)
    qc3.u(params[0], params[1], params[2], position)
    circuits.append(qc3)

    # Step 4: From number 2, add another U(three parameters) with random params in a random position
    qc4 = qc2.copy()
    params = [random.uniform(0, 2 * 3.14159) for _ in range(3)]
    position = random.randint(0, num_qubits - 1)
    qc4.u(params[0], params[1], params[2], position)
    circuits.append(qc4)

    # Step 5: From number 2, remove the U()
    qc5 = qc2.copy()
    params = [random.uniform(0, 2 * 3.14159) for _ in range(3)]
    position = random.randint(0, num_qubits - 1)
    qc5.u(params[0], params[1], params[2], position)
    circuits.append(qc5)

    # Step 6: From number 2, add a CNOT at random
    qc6 = qc2.copy()
    control = random.randint(0, num_qubits - 2)
    target = random.randint(control + 1, num_qubits - 1)
    qc6.cx(control, target)
    circuits.append(qc6)

    # Step 7: From number 6, add another U
    qc7 = qc6.copy()
    params = [random.uniform(0, 2 * 3.14159) for _ in range(3)]
    position = random.randint(0, num_qubits - 1)
    qc7.u(params[0], params[1], params[2], position)
    circuits.append(qc7)

    # Step 8: From number 6, add a CNOT
    qc8 = qc6.copy()
    control = random.randint(0, num_qubits - 2)
    target = random.randint(control + 1, num_qubits - 1)
    qc8.cx(control, target)
    circuits.append(qc8)

    # Step 9: From number 6, add another U
    qc9 = qc6.copy()
    params = [random.uniform(0, 2 * 3.14159) for _ in range(3)]
    position = random.randint(0, num_qubits - 1)
    qc9.u(params[0], params[1], params[2], position)
    circuits.append(qc9)

    # Draw and save the circuits
    for i, qc in enumerate(circuits):
        output_path = os.path.join(output_folder, f'evolution_example_{i}.png')
        qc.draw(output='mpl', initial_state=True, filename=output_path, fold=20, style='bw')
        print(f"Saved circuit {i} as {output_path}")


# Specify the output folder
output_folder = '/Users/bmassacci/main_folder/maastricht/academics/quantum_thesis/images/for_slideshow/'

# Generate and save the quantum circuits
generate_quantum_circuits(output_folder)
