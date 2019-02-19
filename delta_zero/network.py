from abc import ABCMeta, abstractmethod
from datetime import datetime

import os

from keras.models import *
from keras.layers import *
from keras.optimizers import *

import numpy as np

from .utils import labels, dotdict

def_hparams = dotdict(
    channels_in=256,
    input_kernel_size=5,
    residual_kernel_size=3,
    n_residual_layers=5,
    learning_rate=0.2,
    batch_size=64,
    epochs=10
)

class INeuralNetwork(object, metaclass=ABCMeta):

    def __init__(self, name, hparams):
        self.name = name
        self.hparams = hparams
        self.model = self.build()
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                           optimizer=Adam(self.hparams.learning_rate))
        
    @abstractmethod
    def build(self):
        '''
        Builds the network model.
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

    def __init__(self, name='delta_zero'):
        super(ChessNetwork, self).__init__(name, def_hparams)

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

    def train(self, examples):

        state, target_pi, target_v = list(zip(*examples))
        state = np.asarray(state)
        target_pi = np.asarray(target_pi)
        target_v = np.asarray(target_v)
        self.model.fit(x=state,y=[target_pi, target_v],
                       batch_size=self.hparams.batch_size,
                       epochs=self.hparams.epochs)

    def predict(self, state):

        state = np.expand_dims(state, axis=0)
        pi, v = self.model.predict(state)
        return pi[0], v[0]

    def save(self):
        directory = os.path.join(os.path.pardir, 'data', 'models', 'current', 'weights')
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        fn = f'{self.name}_checkpoint.pth.tar'
        fn = os.path.join(directory, fn)
        self.model.save_weights(fn)

    def load(self):
        directory = os.path.join(os.path.pardir, 'data', 'models', 'current', 'weights')
            
        fn = f'{self.name}_checkpoint.pth.tar'
        fn = os.path.join(directory, fn)

        if not os.path.exists(fn):
            raise ValueError(f'Could not load weights for model name: {self.name} : No checkpoint found.')

        self.model.load_weights(fn)
    

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

        
    def _residual_layers(self, X):
        for i in range(self.hparams.n_residual_layers):
            name = f'res{i}'
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
            X = Add(name=f'combine_{name}')([_X, X])
            X = Activation('relu', name=f'relu_{name}_2')(X)
        return X
    
