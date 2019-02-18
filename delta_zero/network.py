from abc import ABCMeta, abstractmethod

from keras.models import *
from keras.optimizers import *

import numpy as np

from utils import labels

def_hparams = dotdict(
    
)

class INeuralNetwork(object, metaclass=ABCMeta):

    def __init__(self, name, hparams):
        self.name = name
        self.hparams = hparams
        self.model = self.build()

    @abstractmethod
    def build(self):
        '''
        Builds and compiles the network model.
        '''
        raise NotImplementedError('build method must be implemented')

    @abstractmethod
    def train(self, examples):
        '''
        Trains the neural network.
        '''
        raise NotImplementedError('train method must be implemented.')

    @abstractmethod
    def predict(self, state):
        '''
        Makes a prediction given a board state.
        '''
        raise NotImplementedError('predict method must be implemented.')

    @abstractmethod
    def save(self):
        '''
        Saves the weights and network configuration to separate files.
        '''
        raise NotImplementedError('save method must be implemented.')

    @abstractmethod
    def load(self):
        '''
        Loads the weights and network configuration.
        '''
        raise NotImplementedError('load method must be implemented.')

class ChessNetwork(INeuralNetwork):

    def __init__(self):
        super(ChessNetwork, self).__init__('delta_zero', def_hparams)

    def build(self):
        X_in = X = Input(shape=(18, 8, 8))
        X = Conv2D(filters=self.hparams.channels_in,
                   kernel_size=self.hparams.input_kernel_size,
                   padding='same',
                   data_format='channels_first',
                   use_bias=False,
                   name='input_conv2d')(X)
        X = BatchNormalization(axis=1, name='input_batchnorm')(X)
        X = Activation('relu', name='input_relu')(X)

        residuals = self._residual_layers(X)
        policy = self._policy_layer(X)
        value = self._value_layer(X, residuals)

        return Model(X_in, [policy, value], name=self.name)

    def _value_layer(self, X, residuals):
        X = Conv2D(filters=4,
                   kernel_size=1,
                   data_format='channels_first',
                   use_bias=False,
                   name='value_conv2d')(residuals)
        X = BatchNormalization(axis=1, name='value_batchnorm')(X)
        X = Activation('relu', name='value_relu')(X)
        X = Flatten(name='value_flatten')(X)
        X = Dense(1, activation='tanh', name='value_out')(X)
        return X

    def _policy_layer(self, X):

        X = Conv2D(filters=self.hparams.channels_in,
                   kernel_size=1,
                   padding='same',
                   data_format='channels_first',
                   use_bias=False,
                   name='policy_conv2d')(X)
        X = BatchNormalization(axis=1, name='policy_batchnorm')(X)
        X = Activation('relu', name='policy_relu')(X)
        X = Flatten(name='policy_flatten')(X)
        X = Dense(len(labels), activation='softmax', name='policy_out')(X)
        return X

        
    def _residual_layers(self, X, name):
        for i in range(self.hparams.n_residual_layers):
            _X = X
            X = Conv2D(filters=self.hparams.channels_in,
                       kernel_size=self.hparams.residual_kernel_size,
                       padding='same',
                       data_format='channels_first',
                       use_bias=False,
                       name=f'conv2d_{name}_1')(X)
            X = BatchNormalization(axis=1, name=f'batchnorm_{name}_1')(X)
            X = Activation('relu', name=f'relu_{name}_1')(X)
            X = Conv2D(filters=self.hparams.channels_in,
                       kernel_size=self.hparams.residual_kernel_size,
                       padding='same',
                       data_format='channels_first',
                       use_bias=False,
                       name=f'conv2d_{name}_2')(X)
            X = BatchNormalization(axis=1, name=f'batchnorm_{name}_2')(X)
            X = Add(name='combine_{name}')([_X, X])
            X = Activation('relu', name=f'relu_{name}_2')(X)
        return X
    
