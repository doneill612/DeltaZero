from abc import ABCMeta, abstractmethod
from datetime import datetime

import os

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add
from keras.layers.core import Activation, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

import numpy as np
import tensorflow as tf

from .utils import labels, dotdict
from .logging import Logger

def_hparams = dotdict(
    filters=256,
    input_kernel_size=3,
    residual_kernel_size=3,
    stride=1,
    fc_size=256,
    n_residual_layers=10,
    learning_rate=0.2,
    batch_size=64,
    epochs=3,
    l2_reg=1e-4
)

logger = Logger.get_logger('ChessNetwork')

class NeuralNetwork(object, metaclass=ABCMeta):
    '''Abstract base class for neural networks in the context of DeltaZero.
    
    This provides a common interface for all potential neural network implementations,
    independent of framework choice.
    '''
    def __init__(self, name, hparams):
        '''Constructs a neural network.
        
        Params
        ------
            name (str): The name of the network. Used for IO operations.
            hparams (dict): The network hyperparameters.
        '''
        self.name = name
        self.hparams = hparams
        self.model = self.build()
        
    @abstractmethod
    def build(self):
        '''
        Builds the network model.
        '''
        raise NotImplementedError('build method must be implemented')

    @abstractmethod
    def train(self, examples):
        '''
        Trains the neural network on the provided examples.

        According the AlphaZero paper, the neural network is responsible
        for taking the board state as input and returning a policy vector
        and scalar value estimation as output. Therefore, training examples
        should be of the form,

            (s; p, v), s := board state, p := policy vector, v := value

        Params
        ------
            examples (array): A numpy.ndarray, or list, of training
                              examples.
        '''
        raise NotImplementedError('train method must be implemented.')

    @abstractmethod
    def predict(self, state):
        '''
        Makes a (policy, value) prediction on given a board state.

        Returns
        -------
            p (array): policy vector, numpy.ndarray
            v (float): value
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

class ChessNetwork(NeuralNetwork):
    '''
    A Keras implementation of the neural network architecture.

    Architectural decisions follow very closely the approach outlined in the paper, with minor deviations.

    The network body consists of:
        - A linear rectified, batch normalized convolutional layer (RBCL) with filter size 256,
          kernel size (3x3) and stride 1.
        - 10 residual layers (as opposed to 19 in the original paper), each residual layer consisting
          of two skip-connected RBCLs, each with filter size 256, kernel size (3x3) and stride 1.

    The network body feeds a policy head and value head.
        - The policy head is very different from the one used by AlphaZero, as AlphaZero represents
          moves in a (73x8x8) matrix stack, while DeltaZero represents moves in a flat-array UCI format.
          DeltaZero's policy head has a single RBCL with filter size 256, kernel size (1x1) and stride 1,
          of which the output is flattened and passed to a linear rectified dense layer of 
          size 1968 (number of possible moves in UCI notation on a chess board).
        - The value head consists of a single RBCL with filter size 4, kernel size (1x1) and stride 1, a
          linear rectified dense layer of size 256, and a tanh dense layer of size 1.
    '''
    def __init__(self, name='delta_zero'):
        self.graph = tf.get_default_graph()
        self.session = tf.Session(graph=self.graph)
        super(ChessNetwork, self).__init__(name, def_hparams)
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                           optimizer=Adam(self.hparams.learning_rate))


    def build(self):
        with self.graph.as_default():
            with self.session.as_default():
                X_in = X = Input(shape=(18, 8, 8))
                X = Conv2D(filters=self.hparams.filters,
                           kernel_size=self.hparams.input_kernel_size,
                           strides=self.hparams.stride,
                           kernel_regularizer=l2(self.hparams.l2_reg),
                           padding='same',
                           data_format='channels_first',
                           use_bias=False,
                           name='input_conv2d')(X)
                X = BatchNormalization(axis=1, name='input_batchnorm')(X)
                X = Activation('relu', name='input_relu')(X)

                for i in range(self.hparams.n_residual_layers):
                    name = f'res{i}'
                    X = self._residual_layer(X, name)
        
                residuals = X
                policy_head = self._policy_layer(X)
                value_head = self._value_layer(X, residuals)

                return Model(X_in, [policy_head, value_head], name=self.name)

    def _value_layer(self, X, residuals):
        with self.graph.as_default():
            with self.session.as_default():
        
                X = Conv2D(filters=4,
                           kernel_size=1,
                           data_format='channels_first',
                           kernel_regularizer=l2(self.hparams.l2_reg),
                           use_bias=False,
                           name='value_conv2d')(residuals)
                X = BatchNormalization(axis=1, name='value_batchnorm')(X)
                X = Activation('relu', name='value_relu')(X)
                X = Flatten(name='value_flatten')(X)
                X = Dense(self.hparams.fc_size, activation='relu', name='value_fc')(X)
                X = Dense(1, activation='tanh', name='value_out')(X)
                return X

    def _policy_layer(self, X):
        with self.graph.as_default():
            with self.session.as_default():
        
                X = Conv2D(filters=self.hparams.filters,
                           kernel_regularizer=l2(self.hparams.l2_reg),
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

        
    def _residual_layer(self, X, name):
        with self.graph.as_default():
            with self.session.as_default():
        
                _X = X
                X = Conv2D(filters=self.hparams.filters,
                           kernel_size=self.hparams.residual_kernel_size,
                           kernel_regularizer=l2(self.hparams.l2_reg),
                           padding='same',
                           data_format='channels_first',
                           use_bias=False,
                           name=f'conv2d_{name}_1')(X)
                X = BatchNormalization(axis=1, name=f'batchnorm_{name}_1')(X)
                X = Activation('relu', name=f'relu_{name}_1')(X)
                X = Conv2D(filters=self.hparams.filters,
                           kernel_size=self.hparams.residual_kernel_size,
                           kernel_regularizer=l2(self.hparams.l2_reg),
                           padding='same',
                           data_format='channels_first',
                           use_bias=False,
                           name=f'conv2d_{name}_2')(X)
                X = BatchNormalization(axis=1, name=f'batchnorm_{name}_2')(X)
                X = Add(name=f'combine_{name}')([_X, X])
                X = Activation('relu', name=f'relu_{name}_2')(X)
            
                return X

    
    def train(self, examples):
        with self.graph.as_default():
            with self.session.as_default():
                state, target_pi, target_v = list(zip(*examples))
                state = np.asarray(state)
                target_pi = np.asarray(target_pi)
                target_v = np.asarray(target_v)
                self.model.fit(x=state,y=[target_pi, target_v],
                               batch_size=self.hparams.batch_size,
                               epochs=self.hparams.epochs)

    def predict(self, state):
        with self.graph.as_default():
            with self.session.as_default():
                state = np.expand_dims(state, axis=0)
                pi, v = self.model.predict(state)
                return pi[0], v[0]

    def save(self, version='nextgen', ckpt=None):
        with self.graph.as_default():
            with self.session.as_default():
                logger.info('Saving model...')
                directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'data',
                                         'models',
                                         f'{version}',
                                         'current', 'weights')

                if not os.path.exists(directory):
                    os.makedirs(directory)

                if ckpt is not None:
                    fn = f'{self.name + "_" + ckpt}_checkpoint.pth.tar'
                else:
                    fn = f'{self.name}_def_checkpoint.pth.tar'
                fn = os.path.join(directory, fn)
                self.model.save_weights(fn)
                logger.info(f'Model saved to {fn}')

    def load(self, version='nextgen', ckpt=None):
        with self.graph.as_default():
            with self.session.as_default():
                logger.info(f'Attempting model load... ckpt: {ckpt}')
                directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'data',
                                         'models',
                                         f'{version}',
                                         'current')
            
                if ckpt is not None:
                    fn = f'{self.name + "_" + ckpt}_checkpoint.pth.tar'
                else:
                    fn = f'{self.name}_def_checkpoint.pth.tar'
                fn = os.path.join(directory, fn)

                if not os.path.exists(fn):
                    ex = f'Could not load weights for model name: {self.name} : No checkpoint found.'
                    logger.fatal(ex)
                    raise ValueError

                self.model.load_weights(fn)
                logger.info(f'Model loaded from {fn}')
    
