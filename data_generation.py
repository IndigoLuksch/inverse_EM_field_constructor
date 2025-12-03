import numpy as np
from tqdm import tqdm
import magpylib as magpy
import random
import numba
from scipy.stats import qmc

import config
print(config.TRAINING_CONFIG.dataset_size)

random.seed(config.RANDOM_SEED)

#magnetic field data generation
class CuboidDataGenerator:
    def __init__(self):
        self.mu_0 = 4 * np.pi * 1e-7

    #@numba.jit(parallel=True, nopython=True)
    def generate_magnets_configs(self):
        sampler = qmc.LatinHypercube(d=5)
        samples = sampler.random(n=config.TRAINING_CONFIG.dataset_size)
        print(samples)

CuboidDataGenerator().generate_magnets_configs()