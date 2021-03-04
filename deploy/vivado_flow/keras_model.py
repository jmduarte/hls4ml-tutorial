"""
 @file   keras_model.py
 @brief  Script for keras model definition
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
########################################################################
# from import

import tensorflow.keras as keras
import tensorflow.keras.models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation
from tensorflow.keras.regularizers import l1
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu


def get_model(name,inputDim,**kwargs):
    if name=='keras_model':
        return get_keras_model(inputDim,**kwargs)
    elif name=='qkeras_model':
        return get_qkeras_model(inputDim,**kwargs)
    else:
        print('ERROR')
        return None

########################################################################
# keras model
########################################################################
def get_keras_model(inputDim,hiddenDim=128,encodeDim=8, batchNorm=True, l1reg=0):
    """
    define the keras model
    the model based on the simple dense auto encoder 
    (128*128*128*128*8*128*128*128*128)
    """
    
    inputLayer = Input(shape=(inputDim,))
    

    kwargs = {'kernel_regularizer': l1(l1reg)}


    h = Dense(hiddenDim,**kwargs)(inputLayer)
    if batchNorm:
        h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(hiddenDim,**kwargs)(h)
    if batchNorm:
        h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(hiddenDim,**kwargs)(h)
    if batchNorm:
        h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(hiddenDim,**kwargs)(h)
    if batchNorm:
        h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(encodeDim,**kwargs)(h)
    if batchNorm:
        h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(hiddenDim,**kwargs)(h)
    if batchNorm:
        h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(hiddenDim,**kwargs)(h)
    if batchNorm:
        h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(hiddenDim,**kwargs)(h)
    if batchNorm:
        h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(hiddenDim,**kwargs)(h)
    if batchNorm:
        h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(inputDim,**kwargs)(h)

    return Model(inputs=inputLayer, outputs=h)

########################################################################
# qkeras model
########################################################################
def get_qkeras_model(inputDim,hiddenDim=128,encodeDim=8,
                     bits=7,intBits=0,
                     reluBits=7,reluIntBits=3,
                     lastBits=7,lastIntBits=3,
                     l1reg=0,batchNorm=True):
    """
    define the keras model
    the model based on the simple dense auto encoder 
    (128*128*128*128*8*128*128*128*128)
    """
    inputLayer = Input(shape=(inputDim,))
    kwargs = {'kernel_quantizer': quantized_bits(bits,intBits,alpha=1),
              'bias_quantizer': quantized_bits(bits,intBits,alpha=1), 
              'kernel_initializer': 'lecun_uniform', 
              'kernel_regularizer': l1(l1reg)
          }

    h = QDense(hiddenDim, **kwargs)(inputLayer)
    if batchNorm:
        h = BatchNormalization()(h)
    h = QActivation(activation=quantized_relu(reluBits,reluIntBits))(h)

    h = QDense(hiddenDim, **kwargs)(h)
    if batchNorm:
        h = BatchNormalization()(h)
    h = QActivation(activation=quantized_relu(reluBits,reluIntBits))(h)

    h = QDense(hiddenDim, **kwargs)(h)
    if batchNorm:
        h = BatchNormalization()(h)
    h = QActivation(activation=quantized_relu(reluBits,reluIntBits))(h)

    h = QDense(hiddenDim, **kwargs)(h)
    if batchNorm:
        h = BatchNormalization()(h)
    h = QActivation(activation=quantized_relu(reluBits,reluIntBits))(h)
    
    h = QDense(encodeDim, **kwargs)(h)
    if batchNorm:
        h = BatchNormalization()(h)
    h = QActivation(activation=quantized_relu(reluBits,reluIntBits))(h)

    h = QDense(hiddenDim, **kwargs)(h)
    if batchNorm:
        h = BatchNormalization()(h)
    h = QActivation(activation=quantized_relu(reluBits,reluIntBits))(h)

    h = QDense(hiddenDim, **kwargs)(h)
    if batchNorm:
        h = BatchNormalization()(h)
    h = QActivation(activation=quantized_relu(reluBits,reluIntBits))(h)

    h = QDense(hiddenDim, **kwargs)(h)
    if batchNorm:
        h = BatchNormalization()(h)
    h = QActivation(activation=quantized_relu(reluBits,reluIntBits))(h)

    h = QDense(hiddenDim, **kwargs)(h)
    if batchNorm:
        h = BatchNormalization()(h)
    h = QActivation(activation=quantized_relu(reluBits,reluIntBits))(h)

    kwargslast = {'kernel_quantizer': quantized_bits(lastBits,lastIntBits,alpha=1),
              'bias_quantizer': quantized_bits(lastBits,lastIntBits,alpha=1), 
              'kernel_initializer': 'lecun_uniform', 
              'kernel_regularizer': l1(l1reg)
          }
    h = QDense(inputDim, **kwargslast)(h)

    return Model(inputs=inputLayer, outputs=h)


    
#########################################################################


from qkeras.utils import _add_supported_quantized_objects
co = {}
_add_supported_quantized_objects(co)    
def load_model(file_path):
    return keras.models.load_model(file_path, custom_objects=co)

    
