from qiskit import QuantumCircuit

with open('/Volumes/SANDISK32_2/final_experiments/24_04_29_21_41_24/evol/final_best_circuit.qasm', 'r') as file:
    qasm_str = file.read()
qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)

# print(qiskit_circuit)
output_path = "/Users/bmassacci/Desktop/circuit.png"
qiskit_circuit.draw(output="mpl", initial_state=True, filename=output_path, fold=20, style="bw")