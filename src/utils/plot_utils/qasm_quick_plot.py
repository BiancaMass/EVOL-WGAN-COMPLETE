import os
from qiskit import QuantumCircuit

round = "4"
folder = "24_05_16_10_05_24"

results_folder = "/Volumes/SANDISK32_2/final_round" + round + "/"
qasm_path = os.path.join(results_folder, folder, "evol", "final_best_circuit.qasm")

with open(qasm_path, 'r') as file:
    qasm_str = file.read()
qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)

# print(qiskit_circuit)

output_path = f"/Users/bmassacci/Desktop/circuit_plots/{round}_{folder}circuit.png"
qiskit_circuit.draw(output="mpl", initial_state=True, filename=output_path, fold=20, style="bw")