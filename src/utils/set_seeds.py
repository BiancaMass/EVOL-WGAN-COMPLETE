import random
import numpy as np
import torch
import tensorflow as tf


def set_seeds(seed):
    print(f'Setting random seed: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
