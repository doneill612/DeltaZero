import os
import random
import time

import numpy as np

import chess.polyglot as book

from .utils import dotdict
from .logging import Logger

def_params = dotdict(
    temp_threshold=10,
    max_hmoves=50
)

logger = Logger.get_logger('ChessAgent')

class ChessAgent(object):
    '''Represents the learning Agent.
    
    The `ChessAgent` will, when in a `ChessEnvironment`,
    make moves with the help of a Monte Carlo Tree Search + Neural Net
    system.

    The `ChessAgent` exposes two methods caled `play()` and `move()`. 
    The former executes a game of self-play and generates supervised training 
    examples to use in training the neural network supporting the MCTS.
    The latter executes a single move in the environment. This is more useful
    for evaluation during which an Agent plays against another opponent.
    '''
    def __init__(self, search_tree, env, params=def_params):
        '''
        Constructs a `ChessAgent`.
        
        Params
        ------
            search_tree (MCTS): a MCTS object instance
            env (ChessEnvironment): the chess environment this Agent will play in
            params (dict): parameter dictionary
        '''
        self.env = env
        self.search_tree = search_tree
        self.params = params

    def play(self, game_name):
        '''
        Makes the agent play itself in a game of chess.
        '''
        examples = []
        step = 0
        turn = 1
        while not self.env.is_game_over:
            step += 1
            
            if step > self.params.max_hmoves:
                self.env.adjudicate()
                continue

            if step % 50 == 0:
                logger.info(f'{step} half moves played in game {game_name}')
            
            c_state = self.env.canonical_board_state
            temperature = step < self.params.temp_threshold

            pi = self.search_tree.pi(self.env, temp=temperature)
            
            examples.append([c_state, turn, pi['pr']])
 
            action = pi['a']
            evaluation = pi['v']
            
            turn *= -1
            self.env.push_action(action)

        res_val = self.env.result_value()
        self.env.to_pgn(self.search_tree.network.name, game_name)
        logger.info(f'Game over - result: {self.env.result_string()}')
        self.reset()
        examples = [[ex[0], ex[2], res_val * ((-1)**(ex[1] != turn))] for ex in examples]
        
        return examples

    def move(self, use_book=False, temp=True):
        if use_book:
            bfp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'data', 'polyglot', 'Performance.bin')

            with book.open_reader(bfp) as reader:
                es = reader.find_all(self.env.board)
                actions = np.asarray([e.move().uci() for e in es])
                action = actions[0]
        else:
            pi = self.search_tree.pi(self.env, temp=temp)
            action = pi['a']
        
        self.env.push_action(action)
        

    def reset(self):
        '''
        Resets the state of this Agent - meaning a total environment and MCTS reset.
        '''
        self.env.reset()
        self.search_tree.reset()

