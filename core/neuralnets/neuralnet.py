from abc import ABCMeta, abstractmethod
import pickle

from core.dzlogging import Logger

config_logger = Logger.get_logger('Config')

class NeuralNetwork(object, metaclass=ABCMeta):
    '''Abstract base class for neural networks in the context of DeltaZero.
    
    This provides a common interface for all potential neural network implementations,
    independent of framework choice.
    '''
    def __init__(self, name, config):
        '''Constructs a neural network.
        
        Params
        ------
            name (str): The name of the network. Used for IO operations.
            config (Config): The network configuration, containing hyperparameters 
                             and metadata.
        '''
        self.name = name
        self.config = config
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
    def train_generator(self, generator, shuffle=True, info_freq=10, write_freq=100):
        '''
        Trains the neural network using examples provided by a generator.

        According the AlphaZero paper, the neural network is responsible
        for taking the board state as input and returning a policy vector
        and scalar value estimation as output. Therefore, training examples
        should be of the form,

            (s; p, v), s := board state, p := policy vector, v := value

        Params
        ------
            generator (generator object) : A generator that yields training
                                           examples sequentially.
            shuffle (bool) : Whether or not to shuffle the examples in the batch
                             before performing gradient descent (recommended).
            info_freq (int) : The frequency in batches for which to receive
                              training statistics
            write_freq (int) : The frequency in batches for which to write
                               training statistics to disk (only applicable for
                               Tensorflow-based implementations)
        '''
        raise NotImplementedError('train_generator method must be implemented.')

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
    def save(self, version):
        '''
        Saves the weights and network configuration.
        '''
        raise NotImplementedError('save method must be implemented.')

    @abstractmethod
    def load(self, version):
        '''
        Loads the weights and network configuration.
        '''
        raise NotImplementedError('load method must be implemented.')

class Config(object):

    def __init__(self,
                 body_filters=32,
                 residual_filters=64,
                 policy_filters=32,
                 value_filters=32,
                 input_kernel_size=3,
                 residual_kernel_size=1,
                 policy_kernel_size=1,
                 value_kernel_size=1,
                 stride=1,
                 fc_size=256,
                 n_residual_layers=1,
                 learning_rate=0.002,
                 lr_decay=1e-5,
                 batch_size=64,
                 epochs=3):
        self.body_filters = body_filters
        self.residual_filters = residual_filters
        self.policy_filters = policy_filters
        self.value_filters = value_filters
        self.input_kernel_size = input_kernel_size
        self.residual_kernel_size = residual_kernel_size
        self.policy_kernel_size = policy_kernel_size
        self.value_kernel_size = value_kernel_size
        self.stride = stride
        self.fc_size = fc_size
        self.n_residual_layers = n_residual_layers
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.epochs = epochs

    def save(self, fp):
        with open(fp, 'wb') as fobj:
            try:
                pickle.dump(self, fobj, protocol=pickle.HIGHEST_PROTOCOL)
            except:
                config_logger.fatal('Couldn\'t save Config object.')
                raise RuntimeError('Config save failed.')

    @staticmethod
    def load(fp):
        with open(fp, 'rb') as fobj:
            try:
                config = pickle.load(fobj)
            except:
                raise RuntimeError('Config load failed.')
            if not isinstance(config, Config):
                logger.fatal('Attempted to load corrupt Config object')
                raise RuntimeError('The supplied filepath does not point '
                                 'to a valid Config object.')
            config_logger.info(f'Config loaded from {fp}.')
            return config

    @staticmethod
    def default_config():
        return Config()
