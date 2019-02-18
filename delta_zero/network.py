from abc import ABCMeta, abstractmethod

from keras.models import *
from keras.optimizers import *
import numpy as np

class INeuralNetwork(object, metaclass=ABCMeta):

    def __init__(self, name):
        self.name = name

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

