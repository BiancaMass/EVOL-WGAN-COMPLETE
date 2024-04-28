import os
import numpy as np
import random
import math
import csv
import torch
from qiskit import QuantumCircuit, QuantumRegister, execute, Aer, IBMQ
# from qiskit.circuit.library import RYGate, RXGate, RZGate, RXXGate, RYYGate, RZZGate
from qiskit.circuit.library import UGate, CXGate

from src.utils.evol_utils.evolutionary_image_utils import crop_images_for_evol
from src.utils.evol_utils.state_embedding import state_embedding, latent_creation

# from src.benchmarking.evol.emd_score_basic_PQC import calculate_benchmark_emd
from src.evolutionary.nets.generator_methods import from_patches_to_image, from_probs_to_pixels
from src.utils.emd_cost_function import emd_scoring_function
from src.utils.evol_utils.evol_plotting import save_tensor
from src.utils.plot_utils.emd_plot_evol import plot_best_fitness


class Qes:
    """
    Evolutionary search for quantum ansatz
    """

    def __init__(self, n_data_qubits, n_ancilla, patch_shape, pixels_per_patch, n_patches,
                 dataloader, evol_batch_size, n_batches, batch_subset, classes,
                 n_children, n_max_evaluations, dtheta, action_weights, multi_action_pb,
                 patch_for_evaluation, device,
                 max_gen_until_change, max_gen_no_improvement, gen_saving_frequency,
                 output_dir, **kwargs):

        """ Initialization the evolutionary algorithm.
        :param n_data_qubits: integer. Number of data qubits for the circuit.
        :param n_ancilla: integer. Number of ancilla qubits for the circuit.
        :param patch_shape: tuple. Dimensions (width, height) of image patches.
        :param pixels_per_patch: int. Number of pixels in one patch.
        :param n_patches: int. number of patches per image. The algorithm generates the ansatz for only one patch.
        :param dataloader: Dataloader. The data used to calculate distance with generated images for the fitness function.
        :param evol_batch_size: int. Number of images in one batch.
        :param n_batches: int. Number of batches of the training dataloader for the alg.
        :param batch_subset: int. How many of the total batches from the train dataloader to use to
            compare the generated patches to the real patches
        :param classes: list. The classes of images for the dataset e.g., [0,1].
        :param n_children: integer. Number of children for each generation.
        :param n_max_evaluations: int. Max number of times a new generated ansatz is evaluated.
        :param dtheta: float. Maximum displacement for the angle parameter in the mutation action.
        :param action_weights: list. Must sum to 100. Weights for choosing Add, Delete, Swap, Mutate.
        :param multi_action_pb: float. Probability to get multiple actions in the same generation.
        :param patch_for_evaluation: int or str. How to select which patch to use for measuring
        the distance between real patches and generated patches. If string, then must be
            'random', and the patches are generated randomly for each image. If int, then indicates
            the top row of the patch. e.g., 3 will take the patch that starts from the 3rd row of
            the image until the row 3 + patch_height.
        :param device: The computation device (e.g., 'cpu', 'cuda').
        :param max_gen_until_change: int. If no improvement for these many generations, dtheta is
              increased to encourage getting out of a local minima.
        :param max_gen_no_improvement: int. When reached these many generations with no improvement, break.
        :param gen_saving_frequency: int. Every how many generations to save output.
        :param output_dir: str. Path to store output files.
        :keyword max_depth: integer. It fixes an upper bound on the quantum circuits depth (the
              length of the critical path (longest sequence of gates)).
        """
        print("Initializing Qes instance")
        # ----- Ansatz Parameters ----- #
        self.n_data_qubits = n_data_qubits
        self.n_ancilla = n_ancilla
        self.n_tot_qubits = n_data_qubits + n_ancilla
        self.n_patches = n_patches
        self.patch_height, self.patch_width = patch_shape[0], patch_shape[1]
        self.pixels_per_patch = pixels_per_patch
        # ----- Input Data Parameters ----- #
        self.dataloader = dataloader
        self.n_batches = n_batches
        self.evol_batch_size = evol_batch_size
        self.number_images_to_compare = batch_subset * evol_batch_size
        self.classes = classes
        # ----- Evolutionary Parameters ----- #
        self.n_children = n_children
        self.n_max_evaluations = n_max_evaluations
        self.dtheta = dtheta
        self.action_weights = action_weights
        self.multi_action_pb = multi_action_pb
        self.patch_for_evaluation = patch_for_evaluation
        self.max_gen_until_change = max_gen_until_change + 1
        self.max_gen_no_improvement = max_gen_no_improvement
        self.n_generations = math.ceil(n_max_evaluations / n_children)
        self.gen_saving_frequency = gen_saving_frequency
        self.sim = Aer.get_backend('statevector_simulator')

        # TODO: for training GPU find a simulator that is GPU compatible
        # if torch.cuda.is_available():
        #     self.sim.set_options(device='GPU')
        self.device = device

        self.output_dir = output_dir

        print('Set up of all the variables complete')

        # -------------------------------------------- #
        # CREATE THE 0-TH INDIVIDUAL (QUANTUM CIRCUIT)
        # Note: state embedding is done separately before calling the evaluation
        qc_0 = QuantumCircuit(QuantumRegister(self.n_tot_qubits, 'qubit'))
        # for qbit in range(self.n_tot_qubits):
        #     qc_0.h(qbit) # Hadamard gates

        # -------------------------------------------- #
        # parameters to store info on CURRENT generation
        self.ind = qc_0  # best individual at the beginning is the vanilla circuit
        # Circuits generated in the current generation from the best qc of the previous one
        self.population = [self.ind]
        self.candidate_sol = None  # Candidate images from the the current generation
        self.fitnesses = None  # Fitness values of the candidate solutions in the current generation
        self.act_choice = None  # Actions taken in the current generation

        # -------------------------------------------- #
        # parameters to store BEST over ALL generations
        self.best_individuals = [qc_0]  # At initialization it is the 0-th circuit
        self.depth = [self.ind.depth()]  # List of circuits depths over the generations
        self.best_solution = []  # Best image produced
        self.best_fitness = []  # Best fitness value found over the generations
        self.best_actions = None  # Best actions taken over the generations

        # -------------------------------------------- #
        # parameters for stopping criteria
        self.no_improvements = 0  # Number of generations without improvements found
        self.fitness_evaluations = 0  # Number of evaluations of the fitness function
        self.current_gen = 0  # Current number of generation in the classical evolutionary algorithm
        # self.emd_benchmark_score = calculate_benchmark_emd()

        self.counting_multi_action = None  # Control over multi actions in the current generations
        self.max_depth = None

        # Add a max_depth argument if provided with additional arguments (kwargs)
        for max_depth in kwargs.values():
            self.max_depth = max_depth
        self.output = None  # All the algorithm data we need to store

        # Preload images from the real dataset to use to calculate the EMD (earth mover distance)
        self.cropped_real_images = crop_images_for_evol(dataloader=self.dataloader,
                                                        patch_position=self.patch_for_evaluation,
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
        - device: '{self.device}'
        - dtheta: {self.dtheta}
        - action_weights: {self.action_weights}
        - multi_action_pb: {self.multi_action_pb}
        - max_gen_until_change: {self.max_gen_until_change}
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
        # print(f"Parent ansatz \n \n {self.ind}")

        for i in range(self.n_children):  # create n_children and apply action to each
            qc = self.ind.copy()  # Current child
            if self.max_depth is not None:  # if user gave a max_depth as input argument
                if qc.depth() >= self.max_depth - 1:  # if current depth is one step away from max
                    counter = 1  # set counter to 1, i.e., only apply one action to the circuit
                    self.counting_multi_action = 0  # to avoid additional actions to be applied
                else:
                    # add 1 extra action with prob. multiaction_prob (so usually 1 action, 2
                    # actions with prob 10% or whatever muliaction_prob is)
                    counter = 1 + self.multiaction().counting_multi_action
            else:
                # add 1 extra action with prob. multiaction_prob (so usually 1 action, 2
                # actions with prob 10% or whatever muliaction_prob is)
                counter = 1 + self.multiaction().counting_multi_action

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

            for j in range(counter):  # go over the selected actions for this one child

                if self.act_choice[j] == 'A':
                    # Chooses 2 locations for the destination qubit(s).
                    # Only one will be used for U, 2 for CNOT
                    position = random.sample([i for i in range(len(qc.qubits))], k=2)
                    # Choose the type of gate (pick an index for the gates list)
                    choice = random.randint(0, len(gate_list) - 1)
                    if choice == 0:  # for the rotation gate u(theta, phi, lambda, qubit)
                        gate_list[choice](angle1, angle2, angle3, position[0])
                    else:
                        gate_list[choice](position[0], position[1])

                elif self.act_choice[j] == 'D':
                    # Pick a position for the gate to remove.
                    if len(qc.data) > 1:
                        position = random.randint(0, len(qc.data) - 1)
                        qc.data.remove(qc.data[position])
                    else:
                        pass
                    # TODO delete
                    # Exclude the the first n_tot_qubits gates (encoding gates) - NOT ANYMORE
                    # if self.n_tot_qubits < len(qc.data) - 1:
                    #     position = random.randint(self.n_tot_qubits, len(qc.data) - 1)
                    #     qc.data.remove(qc.data[position])

                elif self.act_choice[j] == 'S':
                    # Picks a gate and substitutes it with a gate from the family chosen at random

                    # Control if there are enough gates in the circuit to perform a SWAP
                    # TODO check
                    if len(qc.data) > 2:
                        position = random.randint(0, len(qc.data) - 2)
                        remove_ok = True

                    # TODO: delete
                    # Control if there are enough gates in the circuit to perform a SWAP
                    # if len(qc.data) - 1 - self.n_tot_qubits > 0:
                    # Pick a position for the gate to remove and replace
                    # Exclude the the first n_tot_qubits gates (encoding gates)
                    # position = random.randint(self.n_tot_qubits, len(qc.data) - 2)

                    else:  # Handle the case where there are not enough gates to perform a SWAP
                        remove_ok = False

                    if remove_ok:
                        gate_to_remove = qc.data[position][0]  # Get the gate to remove
                        # Choose a new gate to add randomly from the gate dictionary
                        gate_to_add = random.choice(list(gate_dict.values()))
                        # Avoid removing and adding the same gate
                        while gate_to_add.__name__ == gate_to_remove.name:
                            gate_to_add = random.choice(list(gate_dict.values()))
                        if gate_to_add.__name__ == 'CXGate':
                            n_new_qubits = 2
                            gate_to_add = gate_to_add()
                        elif gate_to_add.__name__ == 'UGate':
                            n_new_qubits = 1
                            gate_to_add = gate_to_add(angle1, angle2, angle3)
                        else:
                            print('Error: swap gate not in gate list')
                        # number of qubits of the gate we are removing
                        n_old_qubits = len(qc.data[position][1])
                        # if we are swapping gates with the same amount of qubits, use the same
                        # qubits as the gate we are removing.
                        if n_old_qubits == n_new_qubits:
                            element_to_remove = list(qc.data[position])
                            element_to_remove[0] = gate_to_add  # swap the gates
                            element_to_add = tuple(element_to_remove)
                            qc.data[position] = element_to_add
                        # If gate we are removing has more qubits, pick the qubits from the new
                        # gate as a subset from the original qubits
                        elif n_old_qubits > n_new_qubits:
                            element_to_remove = list(qc.data[position])
                            element_to_remove[0] = gate_to_add
                            element_to_remove[1] = [random.choice(qc.data[position][1])]
                            element_to_add = tuple(element_to_remove)
                            qc.data[position] = element_to_add
                        # If more qubits in the gate we are adding (swapping old gate with)
                        elif n_old_qubits < n_new_qubits:
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
                    # Changes the angle of the selected qubit
                    to_select = 'u'
                    gates_to_mutate = [i for i, gate in enumerate(qc.data[:], start=0)
                                       if gate[0].name == to_select]
                    # TODO: delete
                    # gates_to_mutate = [i for i, gate in enumerate(qc.data[self.n_tot_qubits:],
                    #                                               start=self.n_tot_qubits)
                    #                    if gate[0].name == to_select]

                    if gates_to_mutate:
                        position = random.choice(gates_to_mutate)
                        gate_to_mutate = qc.data[position]
                        # Because U has three parameters
                        angle_to_mutate = random.randint(0, 2)
                        angle_new = gate_to_mutate[0].params[angle_to_mutate] + \
                                    random.uniform(-self.dtheta, self.dtheta)
                        gate_to_mutate[0].params[angle_to_mutate] = angle_new
                    else:  # Skip action if no mutable gates (parameterized) are available
                        pass

                # print(f"circuit after action: {self.act_choice[j]} \n", qc)

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
            qc_with_embedding = state_embedding(qc, self.n_tot_qubits,
                                                latent_creation(self.n_tot_qubits))
            if self.n_patches > 1:
                resulting_image = from_probs_to_pixels(quantum_circuit=qc_with_embedding,
                                                       n_tot_qubits=self.n_tot_qubits,
                                                       n_ancillas=self.n_ancilla,
                                                       sim=self.sim)[:self.pixels_per_patch]

                resulting_image = torch.reshape(torch.from_numpy(resulting_image),
                                                (1, self.patch_height, self.patch_width))
            else:
                resulting_image = from_patches_to_image(quantum_circuit=qc_with_embedding,
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

        :return: instance. Self, with updated fitnesses values.
        """
        self.fitnesses = []  # to store fitness values

        # managing excess candidates (if more candidates than chosen number of children)
        if len(self.candidate_sol) > self.n_children:
            # random.sample samples without replacement. random.choice samples with replacement
            selected_batch = random.sample(self.cropped_real_images, self.number_images_to_compare)
            try:
                self.best_fitness[-1] = emd_scoring_function(
                    real_images_preloaded=selected_batch,
                    num_images_to_compare=self.number_images_to_compare,
                    qc=self.population[0],
                    n_tot_qubits=self.n_tot_qubits,
                    n_ancillas=self.n_ancilla,
                    n_patches=self.n_patches,
                    pixels_per_patch=self.pixels_per_patch,
                    patch_width=self.patch_width,
                    patch_height=self.patch_height,
                    sim=self.sim)
            except Exception as e:
                print(f"An error occurred during fitness function evaluation: {e}")

            self.fitness_evaluations += 1
            del self.candidate_sol[0]
            del self.population[0]

        for i in range(len(self.population)):
            # random.sample samples without replacement. random.choice samples with replacement
            selected_batch = random.sample(self.cropped_real_images, self.number_images_to_compare)
            try:
                self.fitnesses.append(emd_scoring_function(real_images_preloaded=selected_batch,
                                                           num_images_to_compare=self.number_images_to_compare,
                                                           qc=self.population[i],
                                                           n_tot_qubits=self.n_tot_qubits,
                                                           n_ancillas=self.n_ancilla,
                                                           n_patches=self.n_patches,
                                                           pixels_per_patch=self.pixels_per_patch,
                                                           patch_width=self.patch_width,
                                                           patch_height=self.patch_height,
                                                           sim=self.sim))

            except Exception as e:
                print(f"An error occurred during fitness function evaluation: {e}")

            self.fitness_evaluations += 1

        if self.current_gen == 0:
            self.best_fitness.append(self.fitnesses[0])

        return self

    def multiaction(self):
        """
        It permits the individuals to get more actions in the same generations.

        :return: instance. Self, with updated multi-action count (`counting_multi_action`).
        """
        self.counting_multi_action = 0
        rand = random.uniform(0, 1)
        # Increase k time with prob (1-p)*p^3 where p is multi_action_prob
        while rand < self.multi_action_pb:
            self.counting_multi_action += 1
            rand = random.uniform(0, 1)
        return self

    def evolution(self):
        """
        Performs a (1, n_children) evolutionary strategy on quantum circuits to maximize fitness.

        Iterates through generations, applying actions to parent circuits and evaluating offspring fitness.
        Selects the best offspring as the new parent. Adjusts `dtheta` to mitigate local minima and
        adheres to termination criteria based on fitness evaluations or circuit depth.

        :returns: Self, with updated evolutionary process attributes.
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
                if self.fitnesses[index] < self.best_fitness[-1]:
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

                # save every x gens
                if g % self.gen_saving_frequency == 0:
                    image_filename = os.path.join(self.output_dir, f"best_solution_{g}.png")
                    save_tensor(tensor=self.best_solution[-1].squeeze().view(self.patch_height,
                                                                             self.patch_width),
                                filename=image_filename)

                # To reduce probability to get stuck in local minima: change hyper-parameter value
                if self.no_improvements == self.max_gen_until_change:
                    print("Increasing dtheta to exit saddle point")
                    self.dtheta += 0.1
                    # TODO: increase multi action prob
                    # Another way would be to increase self.multi_action_pb
                elif self.no_improvements == 0:  # else reset theta to normal
                    self.dtheta = theta_default
                # Termination criteria
                if self.no_improvements == self.max_gen_no_improvement:
                    print(f"Reached {self.no_improvements} generations with no improvement. "
                          f"Early breaking.")
                    break
                if self.fitness_evaluations == self.n_max_evaluations:
                    print(f"Reached max evaluations ({self.n_max_evaluations}). END.")
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

        # Look if the output directory exists, if not, create it
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # ------------- Save output data file -------------

        self.output = [algo.best_solution, algo.best_individuals[0], algo.best_individuals[-1],
                       algo.depth,
                       algo.best_actions, algo.best_fitness,
                       algo.best_fitness[-1]]

        # Define the headings for the CSV file
        headings = ["Best Solution", "Best Individual - Start", "Best Individual - End",
                    "Depth", "Best Actions", "Best Fitness", "Final Best Fitness"]

        longest_list = max(self.output, key=lambda x: len(x) if isinstance(x, list) else 1)
        num_rows = len(longest_list) if isinstance(longest_list, list) else 1

        columns = []
        for item in self.output:
            if isinstance(item, list):
                # If it's a list, we extend it to match the number of rows
                columns.append(item + [None] * (num_rows - len(item)))
            else:
                # If it's a single value, we repeat it to match the number of rows
                columns.append([item] * num_rows)

        filename_csv = os.path.join(self.output_dir, f"{self.n_children}_"
                                                     f"{self.n_generations}_{self.max_depth}_"
                                                     f"{self.n_patches}_{self.n_tot_qubits}_"
                                                     f"{self.n_ancilla}.csv")

        # Write the data to the CSV file
        with open(filename_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headings)
            writer.writerows(zip(*columns))

        plot_best_fitness(filename_csv, os.path.join(self.output_dir, 'emd_plot.png'))

        # ------------- SAVE CIRCUIT  -------------
        # Quantum circuit as qasm file
        # TODO: or you could remove the whole stripping and re-adding layer in the GAN
        # So basically save without embedding, and remove the part in the GAN wheree you remove
        # the embedding layer
        qc = algo.best_individuals[-1].copy()
        qc_with_embedding = state_embedding(qc, self.n_tot_qubits,
                                            latent_creation(self.n_tot_qubits))
        qasm_best_end = qc_with_embedding.qasm()

        filename_qasm = os.path.join(self.output_dir, f'final_best_circuit.qasm')

        with open(filename_qasm, "w") as file:
            file.write(qasm_best_end)

        # ------------- Save Metadata files -------------

        metadata = {
            "N Data Qubits": self.n_data_qubits,
            "N Ancilla": self.n_ancilla,
            "Patch Shape": (self.patch_height, self.patch_width),
            "Batch Size": self.evol_batch_size,
            "N Children": self.n_children,
            "Max Evaluations": self.n_max_evaluations,
            "DTheta": self.dtheta,
            "Action Weights": self.action_weights,
            "Multi Action Probability": self.multi_action_pb,
            "Max Generations No Improvement": self.max_gen_until_change,
            "Max Generations Until Change": self.max_gen_until_change,
            "Generation Saving Frequency": self.gen_saving_frequency,
            "Output Directory": self.output_dir,
            "Device": self.device,
            "Patch for Evaluation": self.patch_for_evaluation,
            "Number Images to Compare": self.number_images_to_compare,
            "Max Depth": self.max_depth
        }

        metadata_filename_txt = os.path.join(self.output_dir, "metadata.txt")
        metadata_filename_csv = os.path.join(self.output_dir, "metadata.csv")

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

        print(f"Output saved to {filename_csv} and {filename_qasm}")
        print(f"Metadata saved to {metadata_filename_txt} and {metadata_filename_csv}")

        return self
