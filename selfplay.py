import argparse
import os

from random import shuffle

import numpy as np
import ray

from delta_zero.network import ChessNetwork 
from delta_zero.environment import ChessEnvironment
from delta_zero.agent import ChessAgent
from delta_zero.mcts import MCTS
from delta_zero.logging import Logger


logger = Logger.get_logger('selfplay')

ray.init()

@ray.remote
class Runner(object):

    def __init__(self, net_name, iteration, warm_start=None):
        self.net_name = net_name
        self.warm_start = warm_start
        self.iteration = iteration

    def run_selfplay(self):
        '''
        Executes a self-play task.

        Establishes an Agent in an Environment and lets the Agent
        play a full game of chess against itself.

        The resulting training examples are stored to a .npy file in a 
        subdirectory below /data/ with the chosen name of the model.
        '''
        network = ChessNetwork(name=self.net_name)
        try:
            if self.warm_start:
                self.warm_start = str(self.warm_start)
            network.load(ckpt=self.warm_start)
        except ValueError as e:
            logger.verbose(str(e))

        env = ChessEnvironment()

        search_tree = MCTS(network)
        agent = ChessAgent(search_tree, env)

        train_examples = agent.play(game_name=f'{self.net_name}_game{self.iteration+1}')
        train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'delta_zero',
                             'data',
                             net_name,
                             'train')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
            logger.info('Train directory created.')
        fn = os.path.join(train_dir, 'train_set.npy')
        np.save(fn, train_examples)
        logging.info('Game finished - training examples saved.')
    

if __name__ == '__main__':
    Logger.set_log_level('info')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('n_games', type=int,
                        help='The number of games of self-play to train on '
                             'in this session.')
    parser.add_argument('net_name', type=str,
                        help='The name of the network to use. If a saved network '
                             'with this name exists, it is loaded before executing '
                             'self-play.')
    parser.add_argument('warm_start', nargs='?', type=int, help='Network version warm-start')
    
    args = parser.parse_args()
    n_games = args.n_games
    net_name = args.net_name
    warm_start = args.warm_start

    runners = [Runner.remote(net_name=net_name, iteration=i, warm_start=warm_start) for i in range(n_games)]
    ray.get([r.run_selfplay.remote() for r in runners])
    
