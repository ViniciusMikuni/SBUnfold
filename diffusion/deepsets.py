from tensorflow.keras.layers import BatchNormalization, Layer, TimeDistributed
from tensorflow.keras.layers import Dense, Input, ReLU, Masking,Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np



def Resnet(
        inputs,
        end_dim,
        time_embedding,
        num_embed,
        num_layer = 3,
        mlp_dim=128,
):

    
    act = layers.LeakyReLU(alpha=0.01)

    def resnet_dense(input_layer,hidden_size,nlayers=1):
        layer = input_layer
        residual = layers.Dense(hidden_size)(layer)
        for _ in range(nlayers):
            layer = act(layers.Dense(hidden_size,activation=None)(layer))
            layer = layers.Dense(hidden_size,activation=None)(layer)
            layer = layers.Dropout(0.1)(layer)
        return layers.Add()([residual , layer])
    
    embed = act(layers.Dense(mlp_dim)(time_embedding))
    inputs_dense = act(layers.Dense(mlp_dim)(inputs))
    
    residual = act(layers.Dense(mlp_dim)(inputs_dense+embed))
    residual = layers.Dense(mlp_dim)(residual)
    
    layer = residual
    for _ in range(num_layer-1):
        layer =  resnet_dense(layer,mlp_dim)

    layer = act(layers.Dense(2*mlp_dim)(residual+layer))
    outputs = layers.Dense(end_dim)(layer)
    
    return outputs

