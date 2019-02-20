import argparse

from random import shuffle

import numpy as np

from delta_zero.utils import dotdict
from delta_zero.network import ChessNetwork 
from delta_zero.environment import ChessEnvironment
from delta_zero.agent import ChessAgent
from delta_zero.mcts import MCTS

def train(n_sessions, net_name, warm_start=None):
    env = ChessEnvironment()

    network = ChessNetwork(name=net_name)
    try:
        network.load(ckpt=str(warm_start))
    except ValueError as e:
        print(f'WARNING: {e}')

    search_tree = MCTS(network)
    agent = ChessAgent(search_tree, env)

    for s in range(n_sessions):
        train_examples = load_examples(net_name)
        shuffle(train_examples)
        network.train(train_examples)
        network.save(ckpt=str(s))
        
    print('Session complete')    
        

def load_examples(net_name):
    csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'delta_zero',
                           'data',
                           net_name,
                           'train')
    fn = os.path.join(csv_dir, 'train_set.npy')
    return np.load(fn)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('n_sessions', type=int,
                        help='The number of sessions of self-play to execute')
    parser.add_argument('n_games', type=int,
                        help='The number of games of self-play to train on '
                             'in this session.')
    parser.add_argument('warm_start', nargs='?', type=int)
    
    args = parser.parse_args()
    n_games = args.n_games
    n_sessions = args.n_sessions
    warm_start = args.warm_start
    train(n_sessions, n_games, warm_start=warm_start)
    
