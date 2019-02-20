import argparse
import os

from concurrent.futures import ProcessPoolExecutor
from random import shuffle

import numpy as np
import pandas as pd

from delta_zero.utils import dotdict
from delta_zero.network import ChessNetwork 
from delta_zero.environment import ChessEnvironment
from delta_zero.agent import ChessAgent
from delta_zero.mcts import MCTS

    

def selfplay_task(net_name, warm_start):
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
        print(f'WARNING: {e}')

    search_tree = MCTS(network)
    agent = ChessAgent(search_tree, env)

    return agent.play(verbose=False)

def run(n_games, net_name, warm_start=None, max_workers=4):
    '''
    Runs self-play in a thread pool. The results of all the games
    are saved to a .csv file.

    Params
    ------
        n_games (int): the number of games to play (parsed via argparse)
        net_name (str): the network name (loaded if exists)
        warm_start (int): the version of the network to load (optional)
    '''
    train_examples = []
    
    with ProcessPoolExecutor(max_workers=4) as executor:

        print('Process pool established...')

        futures = []
        train_examples = []
        for _ in range(n_games):
            futures.append(executor.submit(selfplay_task, net_name, warm_start))

        print('Playing games...')
        
        for future in futures:
            train_examples.extend(future.result())

        print('All games complete.')

        shuffle(train_examples)
        save_train_examples(train_examples)
    
def save_train_examples(train_examples, fn='test.csv'):
    print('Saving training examples')
    df = pd.DataFrame(_serialize(train_examples), columns=['state', 'policy', 'turn'])
    csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delta_zero', 'data', 'train')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
        print('Train directory created.')
    fn = os.path.join(csv_dir, fn)
    df.to_csv(fn)

def _serialize(examples):
    for ex in examples:
        ex[0], ex[1] = ex[0].tostring(), ex[1].tostring()
    return examples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('n_games', type=int,
                        help='The number of games of self-play to train on '
                             'in this session.')
    parser.add_argument('net_name', type=str,
                        help='The name of the network to use. If a saved network '
                             'with this name exists, it is loaded before executing '
                             'self-play.')

    parser.add_argument('warm_start', nargs='?', type=int, help='Network version warmstrat')
    
    args = parser.parse_args()
    n_games = args.n_games
    net_name = args.net_name
    warm_start = args.warm_start

    run(n_games, net_name, warm_start=warm_start)
