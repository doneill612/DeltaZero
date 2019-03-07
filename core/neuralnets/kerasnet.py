import os

import keras.backend as K

from keras.engine import *
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import SGD

import numpy as np
import tensorflow as tf

from .neuralnet import NeuralNetwork, Config
from core.utils import labels, dotdict, tfsession
from core.dzlogging import Logger

logger = Logger.get_logger('KerasNet')

class KerasNetwork(NeuralNetwork):
    '''
    A Keras implementation of the neural network architecture.

    Architectural decisions follow very closely the approach outlined in the paper, with minor deviations.
    Once a stable architecutre is found it will be described in a markdown file in this directory.
    '''
    def __init__(self, name='delta_zero', config=Config.default_config()):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        config_proto = tf.ConfigProto(gpu_options=gpu_options)
        
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph, config=config_proto)

        K.set_session(self.session)

        super(KerasNetwork, self).__init__(name, config=config)
        log_dir = os.path.join(os.path.pardir,
                               'data',
                               f'{self.name}',
                               'tensorboard',
                               'logdir')
        existed = os.path.exists(log_dir)
        if not existed:
            logger.info('Creating Tensorboard directory...')
            
        self.writer = tf.summary.FileWriter(log_dir, self.graph)
        if not existed:
            self.flush_writer()
        self.compile()

    @tfsession
    def compile(self):
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                           optimizer=SGD(lr=self.config.learning_rate,
                                         decay=self.config.lr_decay,
                                         momentum=0.9,
                                         nesterov=True))
        
    def flush_writer(self, verbose=True):
        self.writer.flush()
        if verbose:
            logger.info('Tensorboard updated.')

    @tfsession
    def build(self):
        with K.name_scope('input_layers'):
            X_in = X = Input(shape=(19, 8, 8))
            X = Conv2D(filters=self.config.body_filters,
                       kernel_size=self.config.input_kernel_size,
                       kernel_initializer='truncated_normal',
                       strides=self.config.stride,
                       padding='same',
                       data_format='channels_first',
                       use_bias=False,
                       name='input_conv2d')(X)
            X = BatchNormalization(axis=1, name='input_batchnorm', epsilon=0.002)(X)
            X = Activation('relu', name='input_relu')(X)
            X = Conv2D(filters=self.config.residual_filters,
                       kernel_size=self.config.input_kernel_size,
                       kernel_initializer='truncated_normal',
                       strides=self.config.stride,
                       padding='same',
                       data_format='channels_first',
                       use_bias=False,
                       name='input_conv2d2')(X)
            X = BatchNormalization(axis=1, name='input_batchnorm2', epsilon=0.002)(X)
            X = Activation('relu', name='input_relu2')(X)
            for i in range(self.config.n_residual_layers):
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
    
        value = Conv2D(filters=self.config.value_filters,
                       kernel_size=self.config.value_kernel_size,
                       kernel_initializer='truncated_normal',
                       data_format='channels_first',
                       use_bias=False,
                       name='value_conv2d')(residuals)
        value = BatchNormalization(axis=1, name='value_batchnorm')(value)
        value = Activation('relu', name='value_relu')(value)
        value = Flatten()(value)
        value = Dense(self.config.fc_size, kernel_initializer='truncated_normal',
                      activation='relu', name='value_fc')(value)
        value = Dense(1, activation='tanh', name='value_out')(value)
        return value

    def _policy_layer(self, residuals):

        policy = Conv2D(filters=self.config.policy_filters,
                        kernel_size=self.config.policy_kernel_size,
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

        _X = X
        X = Conv2D(filters=self.config.residual_filters,
                   kernel_size=self.config.residual_kernel_size,
                   kernel_initializer='truncated_normal',
                   padding='same',
                   data_format='channels_first',
                   use_bias=False,
                   name=f'conv2d_{name}_1')(X)
        X = BatchNormalization(axis=1, name=f'batchnorm_{name}_1', epsilon=0.002)(X)
        X = Activation('relu', name=f'relu_{name}_1')(X)
        X = Conv2D(filters=self.config.residual_filters,
                   kernel_initializer='truncated_normal',
                   kernel_size=self.config.residual_kernel_size,
                   padding='same',
                   data_format='channels_first',
                   use_bias=False,
                   name=f'conv2d_{name}_2')(X)
        X = BatchNormalization(axis=1, name=f'batchnorm_{name}_2', epsilon=0.002)(X)
        X = Add(name=f'combine_{name}')([_X, X])
        X = Activation('relu', name=f'relu_{name}_2')(X)
        return X

    @tfsession
    def train(self, examples):
        es = EarlyStopping('val_loss', min_delta=0.05, patience=3)
        a_s = [np.asarray(ex) for ex in examples]
        ex_np = np.concatenate(a_s)
        state = np.array([s for s in ex_np[:, 0]])
        target_pi = np.array([p for p in ex_np[:, 1]])
        target_v = np.array([v for v in ex_np[:, 2]])
        self.model.fit(x=state,y=[target_pi, target_v],
                       batch_size=self.config.batch_size,
                       epochs=self.config.epochs,
                       shuffle=True, validation_split=0.2,
                       callbacks=[es]) # extra shuffling

    @tfsession
    def train_generator(self, generator,
                        version='current',
                        shuffle=True,
                        info_freq=10,
                        write_freq=100,
                        save_freq=10000):

        if save_freq <= 0:
            save_freq = 10000
        if info_freq <= 0:
            info_freq = 10
        if write_freq <= 0:
            write_freq = 100
            
        epochs = self.config.epochs
        e = 0
        for e in tqdm(range(epochs), desc=f'Epoch {e+1}/{epochs}'):
            for i, (X, y) in enumerate(generator.get(self.config.batch_size)):
                if shuffle:
                    self._shuffle_batch(X, y[0], y[1])
                loss = self.model.train_on_batch(X, y)
                scalar_loss = loss[0]
                policy_loss = loss[1]
                value_loss = loss[2]
                self._add_scalar_summaries(('scalar_loss', scalar_loss),
                                           ('policy_loss', policy_loss),
                                           ('value_loss', value_loss))
                if info_freq > 0 and i % info_freq == 0:
                    logger.info(f'loss: {scalar_loss:.3f}, '
                                f'policy_loss: {policy_loss:.3f}, '
                                f'value_loss: {value_loss:.3f}')
                if write_freq > 0 and i % write_freq == 0:
                    self.flush_writer()
                if i % save_freq == 0:
                        self.save(version=version, ckpt=i+1)
                
    @tfsession
    def predict(self, state):                      
        state = np.expand_dims(state, axis=0)
        pi, v = self.model.predict(state)
        return pi[0], v[0]

    @tfsession
    def save(self, version, ckpt=None):
        logger.info(f'Saving model... checkpoint? : {ckpt}')
        directory = os.path.join(os.path.pardir,
                                 'data',
                                 f'{self.name}',
                                 'models',
                                 f'{version}')

        if not os.path.exists(directory):
            os.makedirs(directory)

        if ckpt is None:
            model_fn = f'{self.name}_checkpoint.h5'
        else:
            model_fn = f'{self.name}_checkpoint_{ckpt}.h5'
        model_fn = os.path.join(directory, model_fn)
        config_fn = os.path.join(directory, f'{self.name}_config.pickle')
        self.model.save(model_fn)
        self.config.save(config_fn)
        logger.info(f'Model saved to {model_fn}')
        self.flush_writer()

    @tfsession
    def load(self, version='nextgen', ckpt=None):
        logger.info(f'Attempting model load... (version: {version}), checkpoint? : {ckpt}')
        directory = os.path.join(os.path.pardir,
                                 'data',
                                 f'{self.name}',
                                 'models',
                                 f'{version}')

        if ckpt is None:
            model_fn = f'{self.name}_checkpoint.h5'
        else:
            model_fn = f'{self.name}_checkpoint_{ckpt}.h5'
        model_fn = os.path.join(directory, model_fn)
        config_fn = os.path.join(directory, f'{self.name}_config.pickle')
        if not os.path.exists(model_fn):
            ex = f'No h5 file was found for model {self.name}.'
            logger.fatal(ex)
            raise RuntimeError('Model load failed.')
        if not os.path.exists(config_fn):
            logger.fatal(f'No configuration file was found for model {self.name}.')
            raise RuntimeError('Model load failed.')
        self.config = Config.load(config_fn)
        try:
            self.model = load_model(model_fn)
            self.compile()
        except ValueError as e:
            logger.fatal('Could not load h5 file for model {self.name}: {e}')
            raise RunetimeError('Model load failed.')
        logger.info(f'Model loaded from {model_fn}')
    

    def _add_scalar_summaries(self, *info):
        for tag, scalar in info:
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=scalar)])
            self.writer.add_summary(summary)

    def _shuffle_batch(self, states, policies, values):
        rs = np.random.get_state()
        np.random.shuffle(states)
        np.random.set_state(rs)
        np.random.shuffle(policies)
        np.random.set_state(rs)
        np.random.shuffle(values)
        np.random.set_state(rs)
