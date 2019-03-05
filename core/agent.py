import os
import random
import time

import numpy as np

import chess.polyglot as book

from .utils import dotdict, labels
from .dzlogging import Logger

def_params = dotdict(
    temp_threshold=30,
    max_hmoves=200,
    n_book_moves=1,
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

    def play(self, game_name, save=True):
        '''
        Makes the agent play itself in a game of chess.
        '''
        examples = []
        step = 0
        turn = 1
        while not self.env.is_game_over:

            step += 1

            if step > self.params.max_hmoves:
                self.env.draw()
                continue

            if step % 50 == 0:
                logger.info(f'{step} half moves played in game {game_name}')
            
            c_state = self.env.canonical_board_state
            temperature = step < self.params.temp_threshold
            
            use_book = step <= self.params.n_book_moves * 2

            if use_book:
                action = self._get_book_move(step)
                pr = np.ones(shape=len(labels)) * np.isin(labels, action, assume_unique=True).astype(np.int32)
                if action is not None:
                    examples.append([c_state, pr, 0.])
            else:
                pi = self.search_tree.pi(self.env, temp=temperature)
                action = pi['a']
                examples.append([c_state, pi['pr'], pi['q']])
            if action is None and use_book:
                pi = self.search_tree.pi(self.env, temp=temperature)
                action = pi['a']
                
                examples.append([c_state, pi['pr'], pi['q']])
            
            self.env.push_action(action)
            
            turn *= -1

        res_val = self.env.result_value()
        if save:
            self.env.to_pgn(self.search_tree.network.name, game_name)
        logger.info(f'Game over - result: {self.env.result_string()}')
        self.reset()

        return examples
        
    def move(self, step, use_book=False):
        temp = step > self.params.temp_threshold
        if use_book:
            action = self._get_book_move(step)
        else:
            pi = self.search_tree.pi(self.env, temp=temp)
            action = pi['a']
        if action is None:
            pi = self.search_tree.pi(self.env, temp=temp)
            action = pi['a']
        self.env.push_action(action)
        

    def reset(self):
        '''
        Resets the state of this Agent - meaning a total environment and MCTS reset.
        '''
        self.env.reset()
        self.search_tree.reset()

    def _get_book_move(self, step):
         bfp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'data', 'polyglot', 'Performance.bin')

         with book.open_reader(bfp) as reader:
             es = reader.find_all(self.env.board)
             actions = np.asarray([e.move().uci() for e in es])
             action = None
             if len(actions) > 0:
                 if step == 1:
                     action = np.random.choice(actions)
                 else:
                     action = actions[0]
             return action
