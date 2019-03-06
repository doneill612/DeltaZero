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
    def train_generator(self, generator, info_freq=10, write_freq=100):

        epochs = self.config.epochs
        e = 0
        for e in tqdm(range(epochs), desc=f'Epoch {e+1}/{epochs}'):
            for i, (X, y) in enumerate(generator):
                loss = self.model.train_on_batch(X, y)
                if info_freq > 0 and i % info_freq == 0:
                    logger.info(f'loss: {loss[0]:.3f}, '
                                f'policy_loss: {loss[1]:.3f}, '
                                f'value_loss: {loss[2]:.3f}')
                if write_freq > 0 and i % write_freq == 0:
                    self.flush_writer()
            
        
    def batch_train(self, X=None, y=None, examples=None, shuffle=True):
        if X is not None and y is not None:
            loss = self._single_batch_gd(X, y, shuffle=shuffle)
            s_summary = tf.Summary(value=[tf.Summary.Value(tag='scalar_loss', simple_value=loss[0])])
            p_summary = tf.Summary(value=[tf.Summary.Value(tag='policy_loss', simple_value=loss[1])])
            v_summary = tf.Summary(value=[tf.Summary.Value(tag='value_loss', simple_value=loss[2])])
            self.writer.add_summary(s_summary)
            self.writer.add_summary(p_summary)
            self.writer.add_summary(v_summary)
            return loss
        if examples is not None:
            self._make_batches_and_gd(examples, shuffle=shuffle)

    def _shuffle_batch(self, states, policies, values):
        rs = np.random.get_state()
        np.random.shuffle(states)
        np.random.set_state(rs)
        np.random.shuffle(policies)
        np.random.set_state(rs)
        np.random.shuffle(values)
        np.random.set_state(rs)

    @tfsession
    def _make_batches_and_gd(self, examples, shuffle=True):
        batch_size = self.config.batch_size
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
                self._shuffle_batch(states, policies, values)
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

    @tfsession
    def _single_batch_gd(self, X, y, shuffle=True):
        if shuffle:
            self._shuffle_batch(X, y[0], y[1])
        return self.model.train_on_batch(X, y)

    @tfsession
    def predict_on_batch(self, X, y, shuffle=True):
        if shuffle:
            rs = np.random.get_state()
            np.random.shuffle(X)
            np.random.set_state(rs)
            np.random.shuffle(y[0])
            np.random.set_state(rs)
            np.random.shuffle(y[1])
            np.random.set_state(rs)
        return self.model.test_on_batch(X, y)

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
    

