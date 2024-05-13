from qiskit import QuantumCircuit

with open('/Users/bmassacci/Desktop/14_2/final_best_circuit.qasm', 'r') as file:
    qasm_str = file.read()
qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)

# print(qiskit_circuit)
output_path = "/Users/bmassacci/Desktop/14_2/circuit.png"
qiskit_circuit.draw(output="mpl", initial_state=True, filename=output_path, fold=20, style="bw")