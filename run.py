#import libraries
import numpy as np
import tensorflow as tf

#import python modules
import data
import config
import model as Model

#---generate data and save to gcloud---
'''
print('Generating data')
generator = data.Dataset()
generator.setup_gcloud()
generator.generate_cubiod_data()
'''

#---load datasets from gcloud---
print('---Loading datasets from GCS---')
dataset_loader = data.Dataset()
dataset_loader.num_points = int((config.AOI_CONFIG['x_dim'] / config.AOI_CONFIG['resolution']) + 1) ** 2

train_dataset = dataset_loader.load_split_datasets(split='train')
val_dataset = dataset_loader.load_split_datasets(split='val')
print('Complete\n\n')

#---create and train model---
print('---Creating model---')
model = Model.create_model()
print('Complete\n\n')

print('---Training model---')

history = Model.train_model(model, train_dataset, val_dataset, initial_lr=config.TRAINING_CONFIG['initial_lr'])
print('Complete\n\n')

print("Script complete")