"""
bash: source .venv-gpu/bin/activate && python test_model.py
--> in same env as model was trained in (tensorflow 2.15)
"""
import tensorflow as tf
import numpy as np
from data import Dataset
import config


#load model
print("\n\n---Loading model and test dataset---")
model_name = "model1.keras"
model = tf.keras.models.load_model(f"models/{model_name}")
print("Model loaded")

#load test dataset from GCS
dataset = Dataset()
test_dataset = dataset.load_split_datasets(split='test')
print("Dataset loaded")

#calculate test steps
test_steps = int(config.DATASET_CONFIG['dataset_size'] * config.DATASET_CONFIG['test_split']) // \
             config.TRAINING_CONFIG['batch_size']

#get predictions and actual values for each data point
print("\n\n---Calculating outputs---")
predictions = []
actual = []

for i, (inputs, labels) in enumerate(test_dataset.take(test_steps)):
    prediction = model.predict(inputs, verbose=0)
    predictions.append(prediction)
    actual.append(labels.numpy())
    if (i + 1) % 10 == 0:
        print(f"{i + 1}/{test_steps} batches")

#concatenate batches
predictions = np.concatenate(predictions, axis=0)
actual = np.concatenate(actual, axis=0)

#calcualte metrics
print("\n\n---Results---")
output_names = ['x position', 'y position', 'dimension a', 'dimension b', 'Mx magnetization', 'My magnetization']
output_ranges = [2*config.AOI_CONFIG['x_dim'],
                 2*config.AOI_CONFIG['x_dim'],
                 config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min'],
                 config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min'],
                 config.MAGNET_CONFIG['M_max'] - config.MAGNET_CONFIG['M_min'],
                 config.MAGNET_CONFIG['M_max'] - config.MAGNET_CONFIG['M_min']]

for i, name in enumerate(output_names):
    mae = np.mean(np.abs(predictions[:, i] - actual[:, i]))
    mae_pc = 100 * mae / output_ranges[i]
    print(f"{name} - MAE: {mae_pc:.6f}% ")

print(f"{'='*60}")

