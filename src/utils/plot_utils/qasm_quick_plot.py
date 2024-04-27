from qiskit import QuantumCircuit

with open('/output/24_04_27_18_01_33/evol/final_best_circuit.qasm', 'r') as file:
    qasm_str = file.read()
qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)

print(qiskit_circuit)