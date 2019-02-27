from abc import ABCMeta, abstractmethod
from datetime import datetime

import os

import numpy as np

from .utils import labels, dotdict
from .logging import Logger

def_hparams = dotdict(
    body_filters=64,
    policy_filters=32,
    input_kernel_size=3,
    residual_kernel_size=3,
    stride=1,
    fc_size=256,
    n_residual_layers=10,
    learning_rate=0.001,
    batch_size=128,
    epochs=10,
    dropout=0.4,
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


import keras.backend as K

from keras.engine import *
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

import tensorflow as tf

class ChessNetwork(NeuralNetwork):
    '''
    A Keras implementation of the neural network architecture.

    Architectural decisions follow very closely the approach outlined in the paper, with minor deviations.
    Once a stable architecutre is found it will be described in a markdown file in this directory.
    '''
    def __init__(self, name='delta_zero'):
 

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9,
                                    allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph, config=config)
        K.set_session(self.session)

        super(ChessNetwork, self).__init__(name, def_hparams)
        with self.graph.as_default():
            with self.session.as_default():
                self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                                   optimizer=Adam(self.hparams.learning_rate))

    def build(self):
        
        with self.graph.as_default():
            with self.session.as_default():
                X_in = X = Input(shape=(19, 8, 8))
                X = Conv2D(filters=self.hparams.body_filters,
                           kernel_size=self.hparams.input_kernel_size,
                           strides=self.hparams.stride,
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
                model = Model(X_in, [policy_head, value_head], name=self.name)
                return model

    def _value_layer(self, X, residuals):
    
        with self.graph.as_default():
            with self.session.as_default():
                X = Conv2D(filters=1,
                           kernel_size=1,
                           data_format='channels_first',
                           use_bias=False,
                           name='value_conv2d')(residuals)
                X = BatchNormalization(axis=1, name='value_batchnorm')(X)
                X = Activation('relu', name='value_relu')(X)
                X = Flatten()(X)
                X = Dense(self.hparams.fc_size, activation='relu', name='value_fc')(X)
                X = Dropout(self.hparams.dropout)(X)
                X = Dense(1, activation='tanh', name='value_out')(X)
                return X

    def _policy_layer(self, X):

        with self.graph.as_default():
            with self.session.as_default():
        
                X = Conv2D(filters=self.hparams.policy_filters,
                           kernel_size=1,
                           padding='same',
                           data_format='channels_first',
                           use_bias=False,
                           name='policy_conv2d')(X)
                X = BatchNormalization(axis=1, name='policy_batchnorm')(X)
                X = Activation('relu', name='policy_relu')(X)
                X = Flatten()(X)
                X = Dense(2048, activation='relu', name='fcp')(X)
                X = Dropout(self.hparams.dropout)(X)
                X = Dense(len(labels), activation='softmax', name='policy_out')(X)
                return X

        
    def _residual_layer(self, X, name):

        with self.graph.as_default():
            with self.session.as_default():
                _X = X
                X = Conv2D(filters=self.hparams.body_filters,
                           kernel_size=self.hparams.residual_kernel_size,
                           kernel_regularizer=l2(self.hparams.l2_reg),
                           padding='same',
                           data_format='channels_first',
                           use_bias=False,
                           name=f'conv2d_{name}_1')(X)
                X = BatchNormalization(axis=1, name=f'batchnorm_{name}_1')(X)
                X = Activation('relu', name=f'relu_{name}_1')(X)
                X = Conv2D(filters=self.hparams.body_filters,
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
                es = EarlyStopping('val_loss', min_delta=0.05, patience=3)
                
                a_s = [np.asarray(ex) for ex in examples]
                ex_np = np.concatenate(a_s)
                state = np.array([s for s in ex_np[:, 0]])
                target_pi = np.array([p for p in ex_np[:, 1]])
                target_v = np.array([v for v in ex_np[:, 2]])
                self.model.fit(x=state,y=[target_pi, target_v],
                               batch_size=self.hparams.batch_size,
                               epochs=self.hparams.epochs,
                               shuffle=True, validation_split=0.2,
                               callbacks=[es]) # extra shuffling
    

    def predict(self, state):
                      
        with self.graph.as_default():
            with self.session.as_default():
                state = np.expand_dims(state, axis=0)
                pi, v = self.model.predict(state)
                return pi[0], v[0]

    def save(self, version):
        
        with self.graph.as_default():
            with self.session.as_default():
                logger.info('Saving model...')
                directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'data',
                                         f'{self.name}',
                                         'models',
                                         f'{version}')

                if not os.path.exists(directory):
                    os.makedirs(directory)
                
                fn = f'{self.name}_checkpoint.pth.tar'
                fn = os.path.join(directory, fn)
                self.model.save_weights(fn)
                logger.info(f'Model saved to {fn}')

    def load(self, version='nextgen'):
        
        with self.graph.as_default():
            with self.session.as_default():
                logger.info(f'Attempting model load... (version: {version})')
                directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'data',
                                         f'{self.name}',
                                         'models',
                                         f'{version}')
            
                fn = f'{self.name}_checkpoint.pth.tar'
                fn = os.path.join(directory, fn)

                if not os.path.exists(fn):
                    ex = f'Could not load weights for model name: {self.name} => No checkpoint found.'
                    logger.warn(ex)
                    raise ValueError

                try:
                    self.model.load_weights(fn)
                except:
                    logger.fatal(f'Could not load weights for model name: {self.name} => Network architecture mismatch.')
                    raise ValueError
                logger.info(f'Model loaded from {fn}')
    
