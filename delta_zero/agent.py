import numpy as np

import time

from .utils import dotdict

def_params = dotdict(
    temp_threshold=10,
    max_hmoves=300
)


class ChessAgent(object):
    '''Represents the learning Agent.
    
    The `ChessAgent` will, when in a `ChessEnvironment`,
    make moves with the help of a Monte Carlo Tree Search + Neural Net
    system.

    The `ChessAgent` exposes a single method caled `play()` which executes
    a game of self-play and generates unsupervised training examples to
    use in training the neural network supporting the MCTS.
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

    def play(self, verbose=True, sleep=False):
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
            
            c_state = self.env.canonical_board_state
            temperature = int(step < self.params.temp_threshold)

            pi = self.search_tree.pi(self.env, temp=temperature)
            
            examples.append([c_state, turn, pi['pr']])
 
            action = pi['a']
            evaluation = pi['v']
            
            turn *= -1
            self.env.push_action(action)

        res_val = self.env.result_value()
        self.env.to_pgn(self.search_tree.network.name, 'test.txt')
        self.reset()
        examples = [[ex[0], ex[2], res_val * ((-1)**(ex[1] != turn))] for ex in examples]
        
        return examples

    def reset(self):
        '''
        Resets the state of this Agent - meaning a total environment and MCTS reset.
        '''
        self.env.reset()
        self.search_tree.reset()
