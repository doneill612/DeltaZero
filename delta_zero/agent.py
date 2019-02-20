import numpy as np

import time

from .utils import dotdict

def_params = dotdict(
    temp_threshold=10,
    max_hmoves=300
)


class ChessAgent(object):

    def __init__(self, search_tree, env, params=def_params):
        self.env = env
        self.search_tree = search_tree
        self.params = params

    def play(self, verbose=False, sleep=False):
        '''
        Makes the agent play itself in a game of chess.
        '''
        examples = []
        step = 0
        turn = 1
        while not self.env.is_game_over:
            step += 1
            
            if step > self.params.max_hmoves:
                if verbose:
                    print('Half move limit reached - adjudicating')
                self.env.adjudicate()
                continue

            if verbose and step % 25 == 0:
                print(f'{step} half moves executed this game...')
            
            c_state = self.env.canonical_board_state
            temperature = int(step < self.params.temp_threshold)

            pi = self.search_tree.pi(self.env.copy(), temp=temperature)
            
            examples.append([c_state, turn, pi['pr']])
 
            action = pi['a']
            evaluation = pi['v']
            
            turn *= -1
            self.env.push_action(action)
            #if verbose:
                #print(f'{"White" if not turn else "Black"} played: {action}\n\n{self.env}')
                #if sleep:
                   # time.sleep(2)

        res_val = self.env.result_value()
        print(f'\nGame result: {self.env.result_string()}')
        self.reset()
        # examples form =>[... [numpy.ndarray, np.ndarray, int] ...]
        examples = [[ex[0], ex[2], res_val * ((-1)**(ex[1] != turn))] for ex in examples]
        
        return examples

    def reset(self):
        self.env.reset()
        self.search_tree.reset()


if __name__ == '__main__':
    for i in range(10):
        print(f'{i}', end='\r')
        time.sleep(0.5)
    print('\n on new line i hope')
