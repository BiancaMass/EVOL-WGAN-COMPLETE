import os
import numpy as np
import random
import math
import csv
import torch
from qiskit import QuantumCircuit, QuantumRegister, execute, Aer, IBMQ
# from qiskit.circuit.library import RYGate, RXGate, RZGate, RXXGate, RYYGate, RZZGate
from qiskit.circuit.library import UGate, CXGate

from src.utils.image_utils.evolutionary_image_utils import crop_images_for_evol

from src.evolutionary.nets.generator_methods import from_patches_to_image, from_probs_to_pixels
from src.utils.emd_cost_function import emd_scoring_function
from src.utils.image_utils.evol_plotting import save_tensor

# TODO: for now hard coded to only consider the first patch.
# TODO: consider later to include either different patches at random, to avoid overfitting and
#  finding a structure for a specific part of the image.

class Qes:
    """
    Evolutionary search for quantum ansatz
    """

    def __init__(self, n_data_qubits, n_ancilla, patch_shape, pixels_per_patch,
                 dataloader, evol_batch_size, n_batches, classes, n_patches,
                 n_children, n_max_evaluations, shots, device,
                 dtheta, action_weights, multi_action_pb, max_gen_no_improvement,
                 output_dir, **kwargs):

        """ Initialization of the population and its settings
        :param n_data_qubits: integer. Number of data qubits for the circuit.
        :param n_ancilla: integer. Number of ancilla qubits for the circuit.
        :param patch_shape: tuple. weight and height of the image.
        :param pixels_per_patch: int. Number of pixels in one patch
        :param dataloader: dataloader. The data to use to calculate fitness with the generated images.
        :param evol_batch_size: # TODO
        :param n_batches: int. Batch size to evaluate a qc (how many times to call it for the
        fitness)
        :param classes: list. The classes of images for the dataset e.g., [0,1] for mnist
        :param n_patches: int. number of patches that the 28x28 image is divided into. The
            evolutionary algorithm then generates the ansatz for only one patch.
        :param n_children: integer. Number of children for each generation.
        :param n_max_evaluations: int. Max number of times a new generated ansatz is evaluated.
        :param shots: integer. Number of executions on the circuit to get the prob. distribution.
        :param dtheta: float. Maximum displacement for the angle parameter in the mutation action.
        :param action_weights: list. Probability to choose between the 4 possible actions. Their
        sum must be 100.
        :param multi_action_pb: float. Probability to get multiple actions in the same generation.
        :param max_gen_no_improvement: int. Number of generations with no improvements after
        which # TODO: what happens?
        :output-dir: str.
        :keyword max_depth: integer. It fixes an upper bound on the quantum circuits depth.
        """
        print("Initializing Qes instance")
        # ----- Ansatz Parameters ----- #
        self.n_data_qubits = n_data_qubits
        self.n_ancilla = n_ancilla
        self.n_tot_qubits = n_data_qubits + n_ancilla
        self.n_patches = n_patches
        self.patch_width, self.patch_height = patch_shape[0], patch_shape[1]
        self.pixels_per_patch = pixels_per_patch
        # ----- Input Data Parameters ----- #
        self.dataloader = dataloader
        self.n_batches = n_batches
        self.evol_batch_size = evol_batch_size
        self.classes = classes
        # ----- Evolutionary Parameters ----- #
        self.n_children = n_children
        self.n_max_evaluations = n_max_evaluations
        self.shots = shots
        self.dtheta = dtheta
        self.action_weights = action_weights
        self.multi_action_pb = multi_action_pb
        self.max_gen_no_improvement = max_gen_no_improvement + 1
        self.n_generations = math.ceil(n_max_evaluations / n_children)
        self.sim = Aer.get_backend('statevector_simulator')

        if torch.cuda.is_available():
            self.sim.set_options(device='GPU')
        self.device = device

        self.output_dir = output_dir

        print('Set up of all the variables complete')

        # -------------------------------------------- #
        # CREATE THE 0-TH INDIVIDUAL (QUANTUM CIRCUIT)
        self.latent_vector_0 = np.random.rand(self.n_tot_qubits)

        qc_0 = QuantumCircuit(QuantumRegister(self.n_tot_qubits, 'qubit'))
        # State embedding: Ry rotation using the latent vector
        for i in range(self.n_tot_qubits):
            qc_0.ry(self.latent_vector_0[i], i)
        # for qbit in range(self.n_tot_qubits):
        #     qc_0.h(qbit) # Hadamard gates

        # -------------------------------------------- #
        # parameters to store info on CURRENT generation
        self.ind = qc_0  # best individual at the beginning is the vanilla circuit
        # Circuits generated in the current generation from the best qc of the previous one
        self.population = [self.ind]
        self.candidate_sol = None  # Candidate images from the the current generation
        self.fitnesses = None # Fitness values of the candidate solutions in the current generation
        self.act_choice = None  # Actions taken in the current generation

        # -------------------------------------------- #
        # parameters to store BEST over ALL generations
        self.best_individuals = [qc_0]  # At initialization it is the 0-th circuit
        self.depth = [self.ind.depth()]  # List of circuits depths over the generations
        self.best_solution = []   # Best image produced
        self.best_fitness = []    # Best fitness value found over the generations
        self.best_actions = None  # Best actions taken over the generations

        self.no_improvements = 0  # Number of generations without improvements found
        self.fitness_evaluations = 0  # Number of evaluations of the fitness function
        self.current_gen = 0  # Current number of generation in the classical evolutionary algorithm
        self.counting_multi_action = None  # Control over multi actions in the current generations
        self.max_depth = None

        # Add a max_depth argument if provided with additional arguments (kwargs)
        for max_depth in kwargs.values():
            self.max_depth = max_depth
        self.output = None  # All the algorithm data we need to store

        # Preload images from the real dataset to use to calculate the EMD (earth mover distance)
        self.cropped_real_images = crop_images_for_evol(dataloader=self.dataloader,
                                                        patch_height=self.patch_height,
                                                        n_batches=self.n_batches,
                                                        device=self.device)

        print('Initial quantum circuit: \n', self.ind)

        print(f"""
        Initialized Qes instance with parameters:
        - n_data_qubits: {self.n_data_qubits}
        - n_ancilla: {self.n_ancilla}
        - n_tot_qubits: {self.n_tot_qubits}
        - n_patches: {self.n_patches}
        - patch_width: {self.patch_width}
        - patch_height: {self.patch_height}
        - pixels_per_patch: {self.pixels_per_patch}
        - n_batches: {self.n_batches}
        - classes: {self.classes}
        - n_children: {self.n_children}
        - n_max_evaluations: {self.n_max_evaluations}
        - n_generations: {self.n_generations}
        - shots: {self.shots}
        - device: '{self.device}'
        - dtheta: {self.dtheta}
        - action_weights: {self.action_weights}
        - multi_action_pb: {self.multi_action_pb}
        - max_gen_no_improvement: {self.max_gen_no_improvement}
        - output directory: {self.output_dir}
        """)

    def action(self):
        """
        Generates n_children of the individual and apply one of the 4 POSSIBLE ACTIONS A,D,S,M
        on each of them. Then the new quantum circuits are stored in 'population'.
        ADD: add a random gate on a random qubit at the end of the parent quantum circuit
        DELETE: delete a random gate in a random position of the parent quantum circuit
        SWAP: Remove a random gate and replace it with a new gate randomly chosen
        MUTATE: Choose a gate and change its angle by a value between [θ-d_θ, θ+d_θ]
        """

        population = []
        print(f"Parent ansatz \n \n {self.ind}")
        # Copy and modify the circuit as many times as the chosen branching factor (`n_children`)
        for i in range(self.n_children):
            qc = self.ind.copy()  # Current child (copy of the current parent)
            if self.max_depth is not None:  # if user gave a max_depth as input argument
                # the depth is the length of the critical path (longest sequence of gates)
                if qc.depth() >= self.max_depth - 1:  # if current depth is one step away from max
                    counter = 1  # set counter to 1, i.e., only apply one action to the circuit
                    self.counting_multi_action = 0  # to avoid additional actions to be applied
                else:
                    # apply *at least* one action
                    counter = self.multiaction().counting_multi_action + 1
            else:
                # apply *at least* one action
                counter = self.multiaction().counting_multi_action + 1

            self.act_choice = random.choices(['A', 'D', 'S', 'M'], weights=self.action_weights,
                                             k=counter)  # outputs a list of k-number of actions
            angle1 = random.random() * 2 * math.pi
            angle2 = random.random() * 2 * math.pi
            angle3 = random.random() * 2 * math.pi
            # TODO: include a choose gate argument to choose which system to use
            # gate_list = [qc.rx, qc.ry, qc.rz, qc.rxx, qc.ryy, qc.rzz]
            # gate_dict = {'rx': RXGate, 'ry': RYGate, 'rz': RZGate,
            #              'rxx': RXXGate, 'ryy': RYYGate, 'rzz': RZZGate}
            gate_list = [qc.u, qc.cx]
            gate_dict = {'UGate': UGate, 'CXGate': CXGate}
            position = 0
            # print('action choice:', self.act_choice, 'for the copy number: ', i)

            for j in range(counter):  # go over the selected actions for this one child

                if self.act_choice[j] == 'A':
                    # Chooses 2 locations for the destination qubit(s).
                    # Only one will be used for U, 2 for CNOT
                    position = random.sample([i for i in range(len(qc.qubits))], k=2)
                    # Choose the type of gate (pick an index for the gates list)
                    choice = random.randint(0, len(gate_list) - 1)
                    if choice == 0:  # for the rotation gate
                        # print(f"Adding a {gate_list[choice]} gate at position: {position[0]}")
                        # u(theta, phi, lambda, qubit)
                        gate_list[choice](angle1, angle2, angle3, position[0])
                    else:
                        # print(f"Adding a {gate_list[choice]} gate at positions: {position}")
                        gate_list[choice](position[0], position[1])

                elif self.act_choice[j] == 'D':
                    # Pick a position for the gate to remove.
                    # Exclude the the first n_tot_qubits gates (encoding gates)
                    if self.n_tot_qubits < len(qc.data) - 1:
                        position = random.randint(self.n_tot_qubits, len(qc.data) - 1)
                        qc.data.remove(qc.data[position])
                    else:
                        pass

                elif self.act_choice[j] == 'S':
                    # Control if there are enough gates in the circuit to perform a SWAP
                    if len(qc.data) - 1 - self.n_tot_qubits > 0:
                        # Pick a position for the gate to swap
                        # Exclude the the first n_tot_qubits gates (encoding gates)
                        position = random.randint(self.n_tot_qubits, len(qc.data) - 2)
                        remove_ok = True
                    else:  # Handle the case where there are not enough gates to swap
                        remove_ok = False

                    if remove_ok:
                        gate_to_remove = qc.data[position][0]  # Get the gate to remove
                        # Choose a new gate to add randomly from the gate dictionary
                        gate_to_add = random.choice(list(gate_dict.values()))
                        # Avoid removing and adding the same gate
                        while gate_to_add.__name__ == gate_to_remove.name:
                            gate_to_add = random.choice(list(gate_dict.values()))
                        if gate_to_add.__name__ == 'CXGate':
                            n_qubits = 2
                            gate_to_add = gate_to_add()
                        elif gate_to_add.__name__ == 'UGate':
                            n_qubits = 1
                            gate_to_add = gate_to_add(angle1, angle2, angle3)
                        else:
                            print('Error: swap gate not in gate list')
                        # number of qubits affected by the gate to be swapped
                        length = len(qc.data[position][1])
                        # if we are swapping gates with the same amount of qubits
                        if length == n_qubits:
                            element_to_remove = list(qc.data[position])
                            element_to_remove[0] = gate_to_add  # swap the gates
                            element_to_add = tuple(element_to_remove)
                            qc.data[position] = element_to_add
                        elif length > n_qubits:
                            element_to_remove = list(qc.data[position])
                            element_to_remove[0] = gate_to_add
                            element_to_remove[1] = [random.choice(qc.data[position][1])]
                            element_to_add = tuple(element_to_remove)
                            qc.data[position] = element_to_add
                        elif length < n_qubits:
                            element_to_remove = list(qc.data[position])
                            element_to_remove[0] = gate_to_add
                            qubits_available = []
                            for q in qc.qubits:
                                if [q] != qc.data[position][1]:
                                    qubits_available.append(q)
                            qubits_ = [qc.data[position][1], random.choice(qubits_available)]
                            random.shuffle(qubits_)
                            element_to_remove[1] = qubits_
                            element_to_add = tuple(element_to_remove)
                            qc.data[position] = element_to_add

                elif self.act_choice[j] == 'M':
                    to_select = 'u'
                    gates_to_mutate = [i for i, gate in enumerate(qc.data[self.n_tot_qubits:],
                                                                  start=self.n_tot_qubits)
                                       if gate[0].name == to_select]

                    if gates_to_mutate:
                        position = random.choice(gates_to_mutate)
                        gate_to_mutate = qc.data[position]
                        angle_to_mutate = random.randint(0, 2)
                        angle_new = gate_to_mutate[0].params[angle_to_mutate] + random.uniform(0,
                                                                                               self.dtheta)
                        gate_to_mutate[0].params[angle_to_mutate] = angle_new
                    else:  # Skip action if no mutable gates (parameterized) are available
                        pass

                # In case of multiactions we are appending more circuits to the population,
                # if you don't want that put the next code line outside of the for loop on counter
            population.append(qc)
        self.population = population
        return self

    def encode(self):
        """
        It transforms a quantum circuit in a string of real values of length 2^N.
        """
        self.candidate_sol = []

        for j in range(len(self.population)):
            qc = self.population[j].copy()
            resulting_image = np.zeros(self.pixels_per_patch)

            if self.n_patches > 1:
                resulting_image = from_probs_to_pixels(
                                            quantum_circuit=qc,
                                            n_tot_qubits=self.n_tot_qubits,
                                            n_ancillas=self.n_ancilla,
                                            sim=self.sim)[:self.pixels_per_patch]

                resulting_image = torch.reshape(
                                            torch.from_numpy(resulting_image),
                                            (1, self.patch_height, self.patch_width))
            else:
                resulting_image = from_patches_to_image(quantum_circuit=qc,
                                                        n_tot_qubits=self.n_tot_qubits,
                                                        n_ancillas=self.n_ancilla,
                                                        n_patches=self.n_patches,
                                                        pixels_per_patch=self.pixels_per_patch,
                                                        patch_width=self.patch_width,
                                                        patch_height=self.patch_height,
                                                        sim=self.sim)

            if self.current_gen == 0:
                self.best_solution.append(resulting_image)

            self.candidate_sol.append(resulting_image)
        return self

    @property
    def fitness(self):
        """Evaluates the fitness of quantum circuits using earth mover distance between real and
        generated flattened images or patches.

        Updates the `fitnesses` list with calculated fitness values for each.

        :return: instance. Self, with updated fitnesses values.
        """
        # Create an empty list to store calculated fitness values
        self.fitnesses = []

        # if there are more candidates than chosen number of children
        # Question: why?
        if len(self.candidate_sol) > self.n_children:
            selected_batch = random.choice(self.cropped_real_images)
            try:
                self.best_fitness[-1] = emd_scoring_function(
                    real_images_preloaded=selected_batch,
                    batch_size=self.evol_batch_size,
                    qc=self.population[0],
                    n_tot_qubits=self.n_tot_qubits,
                    n_ancillas=self.n_ancilla,
                    n_patches=self.n_patches,
                    pixels_per_patch=self.pixels_per_patch,
                    patch_width=self.patch_width,
                    patch_height=self.patch_height,
                    sim=self.sim)
                # elif self.fitness_function == 'critic':
                #     self.best_fitness[-1] = scoring_function(batch_size=self.batch_size,
                #                                              critic=self.critic_net,
                #                                              qc=self.population[0],
                #                                              n_tot_qubits=self.n_tot_qubits,
                #                                              n_ancillas=self.n_ancilla,
                #                                              n_patches=self.n_patches,
                #                                              pixels_per_patch=self.pixels_per_patch,
                #                                              patch_width=self.patch_width,
                #                                              patch_height=self.patch_height,
                #                                              sim=self.sim,
                #                                              device=self.device)
            except Exception as e:
                print(f"An error occurred during fitness function evaluation: {e}")

            self.fitness_evaluations += 1
            del self.candidate_sol[0]
            del self.population[0]

        for i in range(len(self.candidate_sol)):
            selected_batch = random.choice(self.cropped_real_images)
            try:
                self.fitnesses.append(emd_scoring_function(real_images_preloaded=selected_batch,
                                                           batch_size=self.evol_batch_size,
                                                           qc=self.population[i],
                                                           n_tot_qubits=self.n_tot_qubits,
                                                           n_ancillas=self.n_ancilla,
                                                           n_patches=self.n_patches,
                                                           pixels_per_patch=self.pixels_per_patch,
                                                           patch_width=self.patch_width,
                                                           patch_height=self.patch_height,
                                                           sim=self.sim))
                # elif self.fitness_function == 'critic':
                #     self.fitnesses.append(scoring_function(batch_size=self.batch_size,
                #                                            critic=self.critic_net,
                #                                            qc=self.population[i],
                #                                            n_tot_qubits=self.n_tot_qubits,
                #                                            n_ancillas=self.n_ancilla,
                #                                            n_patches=self.n_patches,
                #                                            pixels_per_patch=self.pixels_per_patch,
                #                                            patch_width=self.patch_width,
                #                                            patch_height=self.patch_height,
                #                                            sim=self.sim,
                #                                            device=self.device))

            except Exception as e:
                print(f"An error occurred during fitness function evaluation: {e}")

            self.fitness_evaluations += 1

        if self.current_gen == 0:
            self.best_fitness.append(self.fitnesses[0])

        return self

    def multiaction(self):
        """
        It permits the individuals to get more actions in the same generations. Probability of
        assigning multiple action depends on the parameter `multi_action_pb`.

        :return: instance. Self, with updated multi-action count (`counting_multi_action`).
        """
        self.counting_multi_action = 0
        rand = random.uniform(0, 1)
        # Question: why set it this way? Repeated many times until random happens to be > m.a.probs.
        # Like this, it does not increase counting_multi_action with a probability of
        # multi_action_pb because it does a series of independent runs. I wrote a code to see
        # how much it increased (see script-analysis.py).
        while rand < self.multi_action_pb:
            self.counting_multi_action += 1
            rand = random.uniform(0, 1)
        # print('multiaction counter: ', self.counting_multi_action)
        return self

    def evolution(self):
        """
        Performs a (1, n_children) evolutionary strategy on quantum circuits to maximize fitness.

        Iterates through generations, applying actions to parent circuits and evaluating offspring fitness.
        Selects the best offspring as the new parent. Adjusts `dtheta` to mitigate local minima and
        adheres to termination criteria based on fitness evaluations or circuit depth.

        Updates attributes like `best_individuals`, `depth`, `best_fitness`, `best_solution`, and others
        during the process.

        :returns: Self, with updated evolutionary process attributes.
        :rtype: instance
        """
        self.best_actions = []  # to save in the output file
        action_weights = self.action_weights
        theta_default = self.dtheta
        for g in range(self.n_generations):
            print(f'\ngeneration:{g} of {self.n_generations}')
            if g == 0:
                self.encode().fitness

            else:
                # perform action on parent_ansatz, and then calculate fitness
                self.action().encode().fitness

                index = np.argmin(self.fitnesses)  # index of the best (smallest) fitness value
                # self.fitnesses is the list of fitness values for the current generation
                if self.fitnesses[index] < self.best_fitness[g - 1]:
                    print('improvement found')
                    self.best_individuals.append(self.population[index])
                    self.ind = self.population[index]
                    self.depth.append(self.ind.depth())
                    self.best_fitness.append(self.fitnesses[index])
                    self.best_solution.append(self.candidate_sol[index])
                    for i in range(self.counting_multi_action + 1):
                        self.best_actions.append(self.act_choice[i])

                    self.no_improvements = 0

                else:
                    print('no improvements found')
                    self.no_improvements += 1
                    self.best_individuals.append(self.ind)
                    self.depth.append(self.ind.depth())
                    self.best_fitness.append(self.best_fitness[g - 1])
                    self.best_solution.append(self.best_solution[g - 1])

                # elif self.fitness_function == 'critic':  # with critic goal is to maximize
                #     index = np.argmax(self.fitnesses)  # index of the best (greatest) fitness value
                #     # self.fitnesses is the list of fitness values for the current generation
                #     if self.fitnesses[index] > self.best_fitness[g - 1]:
                #         print('improvement found')
                #         self.best_individuals.append(self.population[index])
                #         self.ind = self.population[index]
                #         self.depth.append(self.ind.depth())
                #         self.best_fitness.append(self.fitnesses[index])
                #         self.best_solution.append(self.candidate_sol[index])
                #         for i in range(self.counting_multi_action + 1):
                #             self.best_actions.append(self.act_choice[i])
                #
                #         self.no_improvements = 0
                #
                #     else:
                #         print('no improvements found')
                #         self.no_improvements += 1
                #         self.best_individuals.append(self.ind)
                #         self.depth.append(self.ind.depth())
                #         self.best_fitness.append(self.best_fitness[g - 1])
                #         self.best_solution.append(self.best_solution[g - 1])

                # Save best image every 10 generations
                if g % 10 == 0:
                    image_filename = os.path.join(self.output_dir, f"best_solution_{g}.png")
                    save_tensor(tensor=self.best_solution[-1].squeeze().view(self.patch_height,
                                                                             self.patch_width),
                                filename=image_filename)

                # To reduce probability to get stuck in local minima: change hyper-parameter value
                if self.no_improvements == self.max_gen_no_improvement:
                    self.dtheta += 0.1
                    # Another way would be to increase self.multi_action_pb
                elif self.no_improvements == 0:
                    self.dtheta = theta_default
                # Termination criteria
                if self.fitness_evaluations == self.n_max_evaluations:
                    break

                if self.max_depth is not None:
                    if self.depth[g] >= self.max_depth:
                        self.action_weights = [0, 20, 0, 80]
                else:
                    self.action_weights = action_weights
            self.current_gen += 1
            print('Number of generations with no improvements: ', self.no_improvements)
            print('best fitness so far: ', self.best_fitness[g])
        print('QES solution: ', self.best_solution[-1])
        return self

    def data(self):
        """ It stores all the data required of the algorithm during the evolution"""
        algo = self.evolution()
        self.output = [algo.best_solution, algo.best_individuals[0], algo.best_individuals[-1],
                       algo.depth,
                       algo.best_actions, algo.best_fitness,
                       algo.best_fitness[-1]]

        # Define the headings for the CSV file
        headings = ["Best Solution", "Best Individual - Start", "Best Individual - End",
                    "Depth", "Best Actions", "Best Fitness", "Final Best Fitness"]

        # Quantum circuit as qasm file
        qasm_best_end = algo.best_individuals[-1].qasm()

        metadata = {
            "N Data Qubits": self.n_data_qubits,
            "N Ancilla": self.n_ancilla,
            "Image Shape": (self.patch_height, self.patch_width),
            "Batch Size": self.evol_batch_size,
            "N Children": self.n_children,
            "Max Evaluations": self.n_max_evaluations,
            "Shots": self.shots,
            "DTheta": self.dtheta,
            "Action Weights": self.action_weights,
            "Multi Action Probability": self.multi_action_pb,
            "Max Generations No Improvement": self.max_gen_no_improvement,
            "Max Depth": self.max_depth,
        }

        # Look if the output directory exists, if not, create it
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filename_csv = os.path.join(self.output_dir, f"{self.n_children}_"
                                                     f"{self.n_generations}_{self.max_depth}_"
                                                     f"{self.n_patches}_{self.n_tot_qubits}_"
                                                     f"{self.n_ancilla}.csv")

        filename_qasm = os.path.join(self.output_dir, f'final_best_circuit.qasm')

        metadata_filename_txt = os.path.join(self.output_dir, "metadata.txt")
        metadata_filename_csv = os.path.join(self.output_dir, "metadata.csv")

        # Write the data to the CSV file
        with open(filename_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headings)
            # Assuming each entry in self.output is iterable and aligned with the headings
            writer.writerow(self.output)

        with open(filename_qasm, "w") as file:
            file.write(qasm_best_end)

        print(f"Output saved to {filename_csv} and {filename_qasm}")

        # Write metadata to the file
        with open(metadata_filename_txt, "w") as f:
            for key, value in metadata.items():
                f.write(f"{key} = {value}\n")

        # Save metadata to CSV file
        with open(metadata_filename_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Variable', 'Value'])  # Write header
            for key, value in metadata.items():
                writer.writerow([key, value])

        print(f"Metadata saved to {metadata_filename_txt} and {metadata_filename_csv}")

        return self
