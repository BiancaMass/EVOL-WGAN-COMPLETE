
# --------------------------------
#     EVOLUTIONARY PARAMETERS
# --------------------------------
N_CHILDREN = 15
M_MAX_EVALUATIONS = 60000
# SHOTS = 2000  # maybe 10000-20000 better for 11 qubits
DTHETA = 0.1
PATCH_FOR_EVALUATION = 'random'
ACTION_WEIGHTS = [70, 5, 5, 20]  # ADD; DELETE; SWAP; MUTATE
MULTI_ACTION_PB = 0.3
MAX_GEN_UNTIL_CHANGE = 20
MAX_GEN_NO_IMPROVEMENT = 250
MAX_DEPTH = 16

GENERATION_SAVING_FREQUENCY = 20

# --------------------------------
#    GAN PARAMETERS (for evol)
# --------------------------------
EVOL_N_BATCHES = 8
EVOL_BATCH_SIZE = 25  # keep high cause distance is calculated on only 1 batch
# How many of the total batches to use to compare the generated patches to the real patches
BATCH_SUBSET = 3  # this means the images used to calculate EMD will be BATCH_SUBSET*EVOL_BATCH_SIZE





