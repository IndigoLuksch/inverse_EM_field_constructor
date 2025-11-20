import os

#paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

#create directories if they do not exist already
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

'''
NOTES 
• all SI units
'''

MAGNET_CONFIG = {
    'height' : 0.1, #m
    'min_side_length' : 0.1, #m
    'max_side_length' : 25, #m
    'min_M' : 1, #T
    'max_M' : 15, #T
}

#area of interest -- area containing desired magnetic field
AOI_CONFIG = {
    'x_start' : 0, #m
    'x_end' : 30, #m
    'y_start' : 0, #m
    'y_end' : 30, #m
}

'''
VIBE
'''

# Model parameters (from paper Section II.D)
MODEL_CONFIG = {
    'name': 'ResNeXt-50',
    'input_shape': (224, 224, 2),  # H_x and H_y components
    'output_dim': 5,  # x, y, a, Mx, My
    'cardinality': 32,  # number of groups
    'base_width': 4,  # channels per group
}

# Training parameters (from paper Section II.D)
TRAINING_CONFIG = {
    'dataset_size': 60000,  # samples (best performance in paper)
    'train_split': 0.6,  # 60%
    'val_split': 0.3,  # 30%
    'test_split': 0.1,  # 10%
    'batch_size': 60,
    'epochs': 100,
    'initial_lr': 0.1,
    'lr_decay_factor': 0.1,
    'lr_decay_epochs': [30, 60],  # decay at epochs 30 and 60
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'loss': 'mse',  # Mean Squared Error (RMSE computed from this)
}

# Data normalization ranges (for target values)
NORMALIZATION = {
    'x': (MAGNET_CONFIG['position_x_min'], MAGNET_CONFIG['position_x_max']),
    'y': (MAGNET_CONFIG['position_y_min'], MAGNET_CONFIG['position_y_max']),
    'a': (MAGNET_CONFIG['side_length_min'], MAGNET_CONFIG['side_length_max']),
    'Mx': (-1.0, 1.0),  # (|M| - 0.5) * cos(phi)
    'My': (-1.0, 1.0),  # (|M| - 0.5) * sin(phi)
}

# Random seed for reproducibility
RANDOM_SEED = 42

'''
VIBE END
'''