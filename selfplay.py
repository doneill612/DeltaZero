import argparse
import multiprocessing as mp
import os

from random import shuffle

import numpy as np
import ray

from delta_zero.network import ChessNetwork 
from delta_zero.environment import ChessEnvironment
from delta_zero.agent import ChessAgent
from delta_zero.mcts import MCTS
from delta_zero.dzlogging import Logger

logger = Logger.get_logger('selfplay')

ray.init()

@ray.remote
class Runner(object):
    '''
    A `ray` actor responsible for executing a single session of self-play.
    '''
    def __init__(self, net_name, iteration, version='nextgen'):
        '''
        Params
        ------
            net_name (str): the name of the network to use
            iteration (int): the game number in this self-play session
            warm_start (int): a warm-start version number to load the network from
        '''
        self.net_name = net_name
        self.version = version
        self.iteration = iteration

    def run_selfplay(self):
        '''
        Executes a self-play task.

        Establishes an Agent in an Environment and lets the Agent
        play a full game of chess against itself.

        The resulting training examples are returned.
        '''
        network = ChessNetwork(name=self.net_name)
        try:
            network.load(version=self.version)
        except ValueError:
            pass
            

        env = ChessEnvironment()

        search_tree = MCTS(network)
        agent = ChessAgent(search_tree, env)

        train_examples = agent.play(game_name=f'{self.net_name}_game{self.iteration+1}')
        logger.info(f'Game complete, generated {len(train_examples)} examples.')
        return train_examples
    

def main():
    '''
    Utilizes all CPU cores to run self-play tasks.
    The results of all the games are aggregated and saved to a training set .npy file.
    '''
    Logger.set_log_level('info')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('net_name', type=str,
                        help='The name of the network to use. If a saved network '
                             'with this name exists, it is loaded before executing '
                             'self-play.')
    parser.add_argument('version', nargs='?', type=str, help='Network version to load - "current" or "nextgen"')
    
    args = parser.parse_args()

    n_games = 1
    
    net_name = args.net_name
    version = args.version

    runners = [Runner.remote(net_name=net_name, iteration=i, version=version) for i in range(n_games)]
    all_examples = []
    train_examples = ray.get([r.run_selfplay.remote() for r in runners])
    for ex in train_examples:
        all_examples.extend(ex)

    all_examples = np.asarray(all_examples)

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
    logger.info(f'Games finished - {len(all_examples)} training examples saved.')
        
if __name__ == '__main__':
    main()
    
