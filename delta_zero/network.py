from abc import ABCMeta, abstractmethod
from datetime import datetime

import os

import numpy as np

from .utils import labels, dotdict
from .dzlogging import Logger

def_hparams = dotdict(
    body_filters=64,
    residual_filters=128,
    policy_filters=64,
    input_kernel_size=3,
    residual_kernel_size=3,
    stride=1,
    fc_size=256,
    n_residual_layers=6,
    learning_rate=.02,
    batch_size=64,
    epochs=3
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
from keras.optimizers import Adam, SGD
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
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'data',
                               f'{self.name}',
                               'tensorboard',
                               'logdir')
        if not os.path.exists(log_dir):
            logger.info('Creating Tensorboard directory...')
        self.writer = tf.summary.FileWriter(log_dir, self.graph)
        self.flush_writer()
        
        with self.graph.as_default():
            with self.session.as_default():
                self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                                   optimizer=SGD(lr=self.hparams.learning_rate,
                                                 decay=1e-4,
                                                 momentum=0.9,
                                                 nesterov=True))

    def flush_writer(self, verbose=True):
        self.writer.flush()
        if verbose:
            logger.info('Tensorboard updated.')
        
    def build(self):
        
        with self.graph.as_default():
            with self.session.as_default():
                with K.name_scope('input_layers'):
                    X_in = X = Input(shape=(19, 8, 8))
                    X = Conv2D(filters=self.hparams.body_filters,
                               kernel_size=self.hparams.input_kernel_size,
                               kernel_initializer='truncated_normal',
                               strides=self.hparams.stride,
                               padding='same',
                               data_format='channels_first',
                               use_bias=False,
                               name='input_conv2d')(X)
                    X = BatchNormalization(axis=1, name='input_batchnorm', epsilon=0.002)(X)
                    X = Activation('relu', name='input_relu')(X)
                    X = Conv2D(filters=self.hparams.residual_filters,
                           kernel_size=self.hparams.input_kernel_size,
                           kernel_initializer='truncated_normal',
                           strides=self.hparams.stride,
                           padding='same',
                           data_format='channels_first',
                           use_bias=False,
                           name='input_conv2d2')(X)
                    X = BatchNormalization(axis=1, name='input_batchnorm2', epsilon=0.002)(X)
                    X = Activation('relu', name='input_relu2')(X)
                for i in range(self.hparams.n_residual_layers):
                    name = f'residual_layer_{i+1}'
                    with K.name_scope(name):
                        X = self._residual_layer(X, name)
                residuals = X
                with K.name_scope('policy_head'):
                    policy_head = self._policy_layer(residuals)
                with K.name_scope('value_head'):
                    value_head = self._value_layer(residuals)
                model = Model(X_in, [policy_head, value_head], name=self.name)
                return model

    def _value_layer(self, residuals):
    
        with self.graph.as_default():
            with self.session.as_default():
                value = Conv2D(filters=32,
                           kernel_size=2,
                           kernel_initializer='truncated_normal',
                           data_format='channels_first',
                           use_bias=False,
                           name='value_conv2d')(residuals)
                value = BatchNormalization(axis=1, name='value_batchnorm')(value)
                value = Activation('relu', name='value_relu')(value)
                value = Flatten()(value)
                value = Dense(self.hparams.fc_size, kernel_initializer='truncated_normal',
                              activation='relu', name='value_fc')(value)
                value = Dense(1, activation='tanh', name='value_out')(value)
                return value

    def _policy_layer(self, residuals):

        with self.graph.as_default():
            with self.session.as_default():
                policy = Conv2D(filters=self.hparams.policy_filters,
                           kernel_size=2,
                           kernel_initializer='truncated_normal',
                           padding='same',
                           data_format='channels_first',
                           use_bias=False,
                           name='policy_conv2d')(residuals)
                policy = BatchNormalization(axis=1, name='policy_batchnorm')(policy)
                policy = Activation('relu', name='policy_relu')(policy)
                policy = Flatten()(policy)
                policy = Dense(len(labels), activation='softmax',
                               kernel_initializer='truncated_normal', name='policy_out')(policy)
                return policy

        
    def _residual_layer(self, X, name):

        with self.graph.as_default():
            with self.session.as_default():
                _X = X
                X = Conv2D(filters=self.hparams.residual_filters,
                           kernel_size=self.hparams.residual_kernel_size,
                           kernel_initializer='truncated_normal',
                           padding='same',
                           data_format='channels_first',
                           use_bias=False,
                           name=f'conv2d_{name}_1')(X)
                X = BatchNormalization(axis=1, name=f'batchnorm_{name}_1', epsilon=0.002)(X)
                X = Activation('relu', name=f'relu_{name}_1')(X)
                X = Conv2D(filters=self.hparams.residual_filters,
                           kernel_initializer='truncated_normal',
                           kernel_size=self.hparams.residual_kernel_size,
                           padding='same',
                           data_format='channels_first',
                           use_bias=False,
                           name=f'conv2d_{name}_2')(X)
                X = BatchNormalization(axis=1, name=f'batchnorm_{name}_2', epsilon=0.002)(X)
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
    

    def batch_train(self, X=None, y=None, examples=None, shuffle=True):
        if X is not None and y is not None:
            with self.graph.as_default():
                with self.session.as_default():
                    loss = self._single_batch_gd(X, y, shuffle=shuffle)
                    s_summary = tf.Summary(value=[tf.Summary.Value(tag='scalar_loss', simple_value=loss[0])])
                    p_summary = tf.Summary(value=[tf.Summary.Value(tag='policy_loss', simple_value=loss[1])])
                    v_summary = tf.Summary(value=[tf.Summary.Value(tag='value_loss', simple_value=loss[2])])
                    self.writer.add_summary(s_summary)
                    self.writer.add_summary(p_summary)
                    self.writer.add_summary(v_summary)
                    return loss
        if examples is not None:
            with self.graph.as_default():
                with self.session.as_default():
                    self._make_batches_and_gd(examples, shuffle=shuffle)

    def _make_batches_and_gd(self, examples, shuffle=True):
        batch_size = self.hparams.batch_size
        n_examples = examples.shape[0]
        for i in range(0, n_examples, batch_size):
            bs = min(i + batch_size, n_examples)
            states = np.empty(shape=(bs, 19, 8, 8))
            policies = np.empty(shape=(bs, 1968))
            values = np.empty(shape=(bs, 1))
            for j in range(i, bs):
                states[j] = examples[j, 0]
                policies[j] = examples[j, 1]
                values[j] = examples[j, 2]
            if shuffle:
                rs = np.random.get_state()
                np.random.shuffle(states)
                np.random.set_state(rs)
                np.random.shuffle(policies)
                np.random.set_state(rs)
                np.random.shuffle(values)
                np.random.set_state(rs)
            loss = self.model.train_on_batch(states, [policies, values])
            s_summary = tf.Summary(value=[tf.Summary.Value(tag='scalar_loss', simple_value=loss[0])])
            p_summary = tf.Summary(value=[tf.Summary.Value(tag='policy_loss', simple_value=loss[1])])
            v_summary = tf.Summary(value=[tf.Summary.Value(tag='value_loss', simple_value=loss[2])])
            self.writer.add_summary(s_summary)
            self.writer.add_summary(p_summary)
            self.writer.add_summary(v_summary)
            logger.info(f'loss: {loss[0]:.3f}, '
                        f'policy_loss: {loss[1]:.3f}, '
                        f'value_loss: {loss[2]:.3f}')

    def _single_batch_gd(self, X, y, shuffle=True):
        if shuffle:
            rs = np.random.get_state()
            np.random.shuffle(X)
            np.random.set_state(rs)
            np.random.shuffle(y[0])
            np.random.set_state(rs)
            np.random.shuffle(y[1])
            np.random.set_state(rs)
        with self.graph.as_default():
            with self.session.as_default():
                return self.model.train_on_batch(X, y)

    def predict_on_batch(self, X, y, shuffle=True):
        if shuffle:
            rs = np.random.get_state()
            np.random.shuffle(X)
            np.random.set_state(rs)
            np.random.shuffle(y[0])
            np.random.set_state(rs)
            np.random.shuffle(y[1])
            np.random.set_state(rs)
        with self.graph.as_default():
            with self.session.as_default():
                return self.model.test_on_batch(X, y)
            
    def predict(self, state):
                      
        with self.graph.as_default():
            with self.session.as_default():
                state = np.expand_dims(state, axis=0)
                pi, v = self.model.predict(state)
                return pi[0], v[0]

    def save(self, version, ckpt=None):
        
        with self.graph.as_default():
            with self.session.as_default():
                logger.info(f'Saving model... checkpoint? : {ckpt}')
                directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'data',
                                         f'{self.name}',
                                         'models',
                                         f'{version}')

                if not os.path.exists(directory):
                    os.makedirs(directory)

                if ckpt is None:
                    fn = f'{self.name}_checkpoint.pth.tar'
                else:
                    fn = f'{self.name}_checkpoint_{ckpt}.pth.tar'
                fn = os.path.join(directory, fn)
                self.model.save_weights(fn)
                logger.info(f'Model saved to {fn}')
                self.flush_writer()

    def load(self, version='nextgen', ckpt=None):
        
        with self.graph.as_default():
            with self.session.as_default():
                logger.info(f'Attempting model load... (version: {version}), checkpoint? : {ckpt}')
                directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'data',
                                         f'{self.name}',
                                         'models',
                                         f'{version}')

                if ckpt is None:
                    fn = f'{self.name}_checkpoint.pth.tar'
                else:
                    fn = f'{self.name}_checkpoint_{ckpt}.pth.tar'
                fn = os.path.join(directory, fn)

                if not os.path.exists(fn):
                    ex = f'Could not load weights for model name: {self.name} => No checkpoint found.'
                    logger.warn(ex)
                    raise ValueError

                try:
                    self.model.load_weights(fn)
                except ValueError as e:
                    logger.fatal(f'Could not load weights for model name: {self.name} => Network architecture mismatch. {e}')
                    raise ValueError
                logger.info(f'Model loaded from {fn}')
    
