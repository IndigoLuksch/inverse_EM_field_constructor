import numpy as np
from tqdm import tqdm
import magpylib as magpy
import random
import numba
from scipy.stats import qmc
import time
import config

random.seed(config.RANDOM_SEED)

#magnetic field data generation
class CuboidDataGenerator:
    '''
    Generates magnetic field data for: 
    • Cuboid magnets with random positions (x, y), side lengths (a, b), and magnetisations (Mx, My)
    • Associated magnetic field maps (H_x, H_y) over the area of interest 
    using latin hypercube sampling and following the specifications in config.py
    '''
    def __init__(self):
        self.mu0 = 4 * np.pi * 1e-7
        self.magnets = np.zeros((config.TRAINING_CONFIG['dataset_size'], 6))  #x, y, a, b, Mx, My

    def generate_magnets(self):
        #---generate magpy magnet collection---
        pbar = tqdm(total=config.TRAINING_CONFIG['dataset_size'], desc="Creating magnets", unit="magnet")
        sampler = qmc.LatinHypercube(d=6)
        samples = sampler.random(n=config.TRAINING_CONFIG['dataset_size'])

        x_samples = qmc.scale(samples[:,0:1], -config.AOI_CONFIG['x_dim'], config.AOI_CONFIG['x_dim']).flatten() #twice AOI size
        y_samples = qmc.scale(samples[:,1:2], -config.AOI_CONFIG['y_dim'], config.AOI_CONFIG['y_dim']).flatten() #twice AOI size
        a_samples = qmc.scale(samples[:,2:3], config.MAGNET_CONFIG['dim_min'], config.MAGNET_CONFIG['dim_max']).flatten()
        b_samples = qmc.scale(samples[:, 3:4], config.MAGNET_CONFIG['dim_min'], config.MAGNET_CONFIG['dim_max']).flatten()
        Mx_samples = qmc.scale(samples[:,4:5], config.MAGNET_CONFIG['M_min'], config.MAGNET_CONFIG['M_max']).flatten()
        My_samples = qmc.scale(samples[:, 5:6], config.MAGNET_CONFIG['M_min'], config.MAGNET_CONFIG['M_max']).flatten()

        magnets = []
        for i in range(config.TRAINING_CONFIG['dataset_size']):
            magnet = magpy.magnet.Cuboid(polarization=(Mx_samples[i], My_samples[i], 0),
                                         dimension=(a_samples[i], b_samples[i], 1),
                                         position=(x_samples[i], y_samples[i], 0))
            magnets.append(magnet)
            pbar.update(1)
        pbar.close()

        #---generate AOI points---
        x = np.arange(-config.AOI_CONFIG['x_dim'] / 2,
                    config.AOI_CONFIG['x_dim'] / 2 + config.AOI_CONFIG['resolution'],
                    config.AOI_CONFIG['resolution'])

        y = np.arange(-config.AOI_CONFIG['y_dim'] / 2,
                      config.AOI_CONFIG['y_dim'] / 2 + config.AOI_CONFIG['resolution'],
                      config.AOI_CONFIG['resolution'])

        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        #---calculate magnetic field at each AOI point
        pbar = tqdm(total = config.TRAINING_CONFIG['dataset_size'], desc='Calculating H fields')
        H = np.zeros((config.TRAINING_CONFIG['dataset_size'], #number of magnets
                      points.shape[0], #number of points
                      2)) #Hx, Hy
        for i in range( config.TRAINING_CONFIG['dataset_size'] ):
            H[i] = magpy.getH(magnets[i], points)[:, :2]  #only store Hx, Hy
            pbar.update(1)
        pbar.close()

    #def generate_fields(self, magnets):

instance = CuboidDataGenerator()
instance.generate_magnets()