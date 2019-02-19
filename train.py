import argparse
from random import shuffle

import numpy as np
import tqdm

from delta_zero.utils import dotdict
from delta_zero.network import ChessNetwork 
from delta_zero.environment import ChessEnvironment
from delta_zero.agent import ChessAgent
from delta_zero.mcts import MCTS

def train(n_sessions, n_games):
    env = ChessEnvironment()

    network = ChessNetwork()
    try:
        network.load()
    except ValueError as e:
        print(f'WARNING: {e}')

    search_tree = MCTS(network)
    agent = ChessAgent(search_tree, env)

    for _ in tqdm(range(n_sessions), ascii=True, desc='Running sessions', position=0):
        train_examples = []

        for __ in tqdm(range(n_games), ascii=True, desc='Playing games', position=1):
            train_examples.append(agent.play())

        shuffle(train_examples)
        network.train(train_examples)

    network.save()
    print('Session complete')    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_games', type=int,
                        help='The number of games of self-play to train on '
                             'in this session.')
    parser.add_argument('n_sessions', type=int,
                        help='The number of sessions of self-play to execute')
    args = parser.parse_args()
    n_games = args.n_games
    n_sessions = args.n_sessions
    train(n_sessions, n_games)
    
