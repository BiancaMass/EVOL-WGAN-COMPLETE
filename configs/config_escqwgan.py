
# --------------------------------
#     EVOLUTIONARY PARAMETERS
# --------------------------------
N_CHILDREN = 10
M_MAX_EVALUATIONS = 1000
SHOTS = 2000  # maybe 10000-20000 better for 11 qubits
DTHETA = 0.1
ACTION_WEIGHTS = [60, 10, 10, 20]  # ADD; DELETE; SWAP; MUTATE
MULTI_ACTION_PB = 0.1
MAX_GEN_NO_IMPROVEMENT = 10
MAX_DEPTH = 10

# --------------------------------
#    GAN PARAMETERS (for evol)
# --------------------------------
EVOL_BATCH_SIZE = 128  # keep high cause distance is calculated on only 1 batch
LR_G = 0.01  # learning rate for the generator
# Betas, initial decay rate for the Adam optimizer
# Check: if these values are appropriate
B1 = 0    # Beta1, the exponential decay rate for the 1st moment estimates. Default would be 0.9
B2 = 0.9  # Beta2, the exponential decay rate for the 2nd moment estimates. Default would be 0.999
LAMBDA_GP = 10  # Coefficient for the gradient penalty






