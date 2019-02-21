import argparse
import os

from concurrent.futures import ProcessPoolExecutor
from random import shuffle

import numpy as np

from delta_zero.network import ChessNetwork 
from delta_zero.environment import ChessEnvironment
from delta_zero.agent import ChessAgent
from delta_zero.mcts import MCTS
from delta_zero.logging import Logger

Logger.set_log_level('info')
logger = Logger.get_logger('selfplay')

def run(n_games, net_name, warm_start=None, max_workers=4):
    '''
    Runs self-play in a process pool. The results of all the games
    are saved to a .npy file.

    Params
    ------
        n_games (int): the number of games to play (parsed via argparse)
        net_name (str): the network name (loaded if exists)
        warm_start (int): the version of the network to load (optional)
    '''
    train_examples = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        logger.info('Process pool established...')

        futures = []
        train_examples = []
        for i in range(n_games):
            futures.append(executor.submit(selfplay_task, net_name, warm_start, i))

        logger.info(f'Playing {n_games} games...')
        
        for i, future in enumerate(futures):
            train_examples.extend(future.result())
            logger.info(f'{i+1} game{"" if i <= 1 else "s"} finished...')

        logger.info(f'All games complete. Generated {len(train_examples)} examples.')

        shuffle(train_examples)
        save_train_examples(np.asarray(train_examples), net_name)

def selfplay_task(net_name, warm_start, i):
    '''
    Executes self-play task.

    Establishes an Agent in an Environment and lets the Agent
    play a full game of chess against itself.

    Params
    ------
        n_games (int): the number of games to play (parsed via argparse)
        net_name (str): the network name (loaded if exists)
        warm_start (int): the version of the network to load (optional)
    '''
    env = ChessEnvironment()

    network = ChessNetwork(name=net_name)
    try:
        if warm_start:
            warm_start = str(warm_start)
        network.load(ckpt=warm_start)
    except ValueError as e:
        logger.warn(e)

    search_tree = MCTS(network)
    agent = ChessAgent(search_tree, env)
 
    return agent.play()

    
def save_train_examples(train_examples, net_name, fn='train_set.npy'):
    train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'delta_zero',
                             'data',
                             net_name,
                             'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        logger.info('Train directory created.')
    fn = os.path.join(train_dir, fn)
    np.save(fn, train_examples)
    
if __name__ == '__main__':
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

    run(n_games, net_name, warm_start=warm_start)
