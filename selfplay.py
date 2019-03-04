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
    def __init__(self, net_name, iteration, version='current'):
        '''
        Params
        ------
            net_name (str): the name of the network to use
            iteration (int): the game number in this self-play session
            
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
        exs = agent.play(game_name=f'{self.net_name}_game{self.iteration+1}', save=False)
        
        return exs
    
def main(net_name, n_sessions, gps, ckpt=None):
    '''
    Utilizes all CPU cores to run self-play tasks.
    '''
    for j in range(n_sessions):
        exs = []
        runners = [Runner.remote(net_name=net_name, iteration=i) for i in range(gps)]
        exs.extend(ray.get([r.run_selfplay.remote() for r in runners]))
        exs = np.squeeze(np.asarray(exs))
        if len(exs.shape) > 2:
            exs = exs.reshape((exs.shape[0] * exs.shape[1], exs.shape[2]))
        logger.info(f'Session examples extracted: {exs.shape[0]}')
        network = ChessNetwork(net_name)
        try:
            network.load(version='current')
        except:
            logger.warn('No current version found, starting fresh.')
        network.batch_train(examples=exs)
        network.save(version='current', ckpt=exs.shape[0])
    
  
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--netname', dest='net_name', type=str,
                        help='The name of the network to use. If a saved network '
                             'with this name exists, it is loaded before executing '
                             'self-play.')
    parser.add_argument('--sessions', dest='nsessions', default=1, nargs='?', type=int,
                        help='The number of self-play sessions to execute.')
    parser.add_argument('--gps', dest='gps', default=1, nargs='?', type=int,
                        help='The number of games per session of self-play to execute.')
    parser.add_argument('--checkpoint', dest='ckpt', default=None, nargs='?', type=int,
                        help='A checkpoint integer indicating which specific iteration of '
                             'the network to load.')
    args = parser.parse_args()
    net_name = args.net_name
    ckpt = args.ckpt
    nsessions = args.nsessions
    gps = args.gps
    
    main(net_name, nsessions, gps, ckpt=ckpt)
    
