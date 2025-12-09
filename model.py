import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import config

#---edit model architecture---
def edit_resnet50(input_shape=config.MODEL_CONFIG['input_shape'], output_dim=config.MODEL_CONFIG['output_dim']):
    base_model = ResNet50(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling='avg' #use global average pooling rather than flattening (flattening forces model to train on oddly shaped vectors that do not capture the shape of the data)
    )

    output = layers.Dense(output_dim, activation=None)(base_model.output) #create output layer of right shape, linear activation

    model = models.Model(inputs=base_model.input, outputs=output)

    print("Model created")
    return model

if __name__ == '__main__':
    edit_resnet50()