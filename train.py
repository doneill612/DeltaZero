import argparse

from random import shuffle

import numpy as np

from delta_zero.utils import dotdict
from delta_zero.network import ChessNetwork 
from delta_zero.environment import ChessEnvironment
from delta_zero.agent import ChessAgent
from delta_zero.mcts import MCTS

def train(n_sessions, n_games, warm_start=None):
    env = ChessEnvironment()

    network = ChessNetwork(name='delta_zero3')
    try:
        network.load(ckpt=str(warm_start))
    except ValueError as e:
        print(f'WARNING: {e}')

    search_tree = MCTS(network)
    agent = ChessAgent(search_tree, env)

    for s in range(n_sessions):
        train_examples = []

        for g in range(n_games):
            print(f'Playing game {g+1}...')
            train_examples.extend(agent.play())

        shuffle(train_examples)
        network.train(train_examples)
        network.save(ckpt=str(s))
        
    print('Session complete')    
        

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
    
