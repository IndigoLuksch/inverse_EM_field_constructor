"""
ResNeXt-50 (32x4d) implementation in TensorFlow
Based on: Xie et al., "Aggregated Residual Transformations for Deep Neural Networks", CVPR 2017
Adapted for magnetic field inverse design regression task
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple

from config import MODEL_CONFIG

'''
BELOW IS MOSTLY VIBE CODED
'''

def grouped_convolution(inputs, filters: int, cardinality: int, strides: Tuple[int, int] = (1, 1), name: str = None):
    """
    Optimized Grouped Convolution using native TensorFlow implementation.
    """
    # Native groups support handles splitting and convolving efficiently
    return layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=strides,
        padding='same',
        groups=cardinality,  # Native implementation
        use_bias=False,      # Usually False before BN
        kernel_initializer='he_normal',
        name=f'{name}_grouped_conv' if name else None
    )(inputs)


def resnext_block(inputs,
                  filters: int,
                  cardinality: int = 32,
                  base_width: int = 4,
                  strides: Tuple[int, int] = (1, 1),
                  name: str = None):
    width = base_width * cardinality

    # Shortcut
    if strides != (1, 1) or inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='same',
                                 use_bias=False, kernel_initializer='he_normal',
                                 name=f'{name}_shortcut_conv')(inputs)
        shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)
    else:
        shortcut = inputs

    # 1x1 Bottleneck
    x = layers.Conv2D(width, (1, 1), strides=(1, 1), padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      name=f'{name}_conv1')(inputs)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.ReLU(name=f'{name}_relu1')(x)

    # 3x3 Grouped Conv (Optimized)
    x = grouped_convolution(x, width, cardinality, strides=strides, name=f'{name}_conv2')
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    x = layers.ReLU(name=f'{name}_relu2')(x)

    # 1x1 Expansion
    x = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      name=f'{name}_conv3')(x)
    x = layers.BatchNormalization(name=f'{name}_bn3')(x)

    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.ReLU(name=f'{name}_relu')(x)

    return x


def build_resnext50_32x4d(input_shape: Tuple[int, int, int] = (224, 224, 2),
                          output_dim: int = 5,
                          cardinality: int = 32,
                          base_width: int = 4):
    """
    Build ResNeXt-50 (32x4d) model for magnetic field inverse design.

    Architecture follows ResNet-50 but uses grouped convolutions:
    - Initial conv + max pool
    - Stage 1: 3 blocks, 64 filters
    - Stage 2: 4 blocks, 128 filters
    - Stage 3: 6 blocks, 256 filters
    - Stage 4: 3 blocks, 512 filters
    - Global average pooling
    - Fully connected layer (5 outputs for regression)

    Args:
        input_shape: Input shape (height, width, channels)
        output_dim: Number of output values (5 for x, y, a, Mx, My)
        cardinality: Number of groups for grouped convolution
        base_width: Base width for each group

    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape, name='input_magnetic_field')

    # Initial convolution
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    # Stage 1: 3 blocks (64 filters, no downsampling)
    x = resnext_block(x, 256, cardinality, base_width, strides=(1, 1), name='stage1_block1')
    x = resnext_block(x, 256, cardinality, base_width, strides=(1, 1), name='stage1_block2')
    x = resnext_block(x, 256, cardinality, base_width, strides=(1, 1), name='stage1_block3')

    # Stage 2: 4 blocks (128 filters, downsample first block)
    x = resnext_block(x, 512, cardinality, base_width, strides=(2, 2), name='stage2_block1')
    x = resnext_block(x, 512, cardinality, base_width, strides=(1, 1), name='stage2_block2')
    x = resnext_block(x, 512, cardinality, base_width, strides=(1, 1), name='stage2_block3')
    x = resnext_block(x, 512, cardinality, base_width, strides=(1, 1), name='stage2_block4')

    # Stage 3: 6 blocks (256 filters, downsample first block)
    x = resnext_block(x, 1024, cardinality, base_width, strides=(2, 2), name='stage3_block1')
    x = resnext_block(x, 1024, cardinality, base_width, strides=(1, 1), name='stage3_block2')
    x = resnext_block(x, 1024, cardinality, base_width, strides=(1, 1), name='stage3_block3')
    x = resnext_block(x, 1024, cardinality, base_width, strides=(1, 1), name='stage3_block4')
    x = resnext_block(x, 1024, cardinality, base_width, strides=(1, 1), name='stage3_block5')
    x = resnext_block(x, 1024, cardinality, base_width, strides=(1, 1), name='stage3_block6')

    # Stage 4: 3 blocks (512 filters, downsample first block)
    x = resnext_block(x, 2048, cardinality, base_width, strides=(2, 2), name='stage4_block1')
    x = resnext_block(x, 2048, cardinality, base_width, strides=(1, 1), name='stage4_block2')
    x = resnext_block(x, 2048, cardinality, base_width, strides=(1, 1), name='stage4_block3')

    # Global average pooling
    x = layers.GlobalAveragePooling2D(name='global_avgpool')(x)

    # Fully connected layer for regression
    outputs = layers.Dense(output_dim, activation='linear', name='output_magnet_params')(x)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='ResNeXt-50-32x4d')

    return model


def build_model():
    """
    Build and return ResNeXt-50 model with configuration from config.py
    """
    model = build_resnext50_32x4d(
        input_shape=MODEL_CONFIG['input_shape'],
        output_dim=MODEL_CONFIG['output_dim'],
        cardinality=MODEL_CONFIG['cardinality'],
        base_width=MODEL_CONFIG['base_width']
    )
    return model


if __name__ == '__main__':
    # Test model building
    print("Building ResNeXt-50 (32x4d) model...")
    model = build_model()

    print("\nModel Summary:")
    model.summary()

    print(f"\nTotal parameters: {model.count_params():,}")

    # Test forward pass
    import numpy as np

    dummy_input = np.random.randn(1, 224, 224, 2).astype(np.float32)
    output = model(dummy_input, training=False)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output values: {output.numpy()[0]}")
