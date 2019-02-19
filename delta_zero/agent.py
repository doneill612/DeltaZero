import numpy as np

from .utils import dotdict

def_params = dotdict(
    temp_threshold=10
)

class ChessAgent(object):

    def __init__(self, search_tree, env, params=def_params):
        self.env = env
        self.search_tree = search_tree
        self.params = params
        
    def play(self):
        '''
        Makes the agent play itself in a game of chess.
        '''
        examples = []
        step = 0
        turn = 1
        while not self.env.is_game_over:
            step += 1
            
            c_state = self.env.canonical_board_state
            temperature = int(step < self.params.temp_threshold)

            pi = self.search_tree.pi(self.env.copy(), temp=temperature)
            examples.append([c_state, turn, pi['pr']])

            action = pi['a']
            turn *= -1
        
            self.env.push_action(action)
            

        res_val = self.env.result_value()
        self.env.reset()
        
        
        return [(ex[0], ex[2], res_val * ((-1)**(ex[1] != turn)))
                for ex in examples]
