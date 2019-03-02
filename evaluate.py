import argparse
import multiprocessing as mp
import random

import numpy as np
import ray

from delta_zero.network import ChessNetwork 
from delta_zero.environment import ChessEnvironment
from delta_zero.agent import ChessAgent
from delta_zero.mcts import MCTS
from delta_zero.logging import Logger
from delta_zero.dzutils import dotdict

logger = Logger.get_logger('evaluation')

ray.init()

@ray.remote
class Runner(object):

    def __init__(self, net_name, iteration, cur_white=True):
        self.net_name = net_name
        self.iteration = iteration
        self.game_name = f'Game {self.iteration + 1}'
        self.cur_white = cur_white

    def run_evaluation(self):
        current_network = ChessNetwork(self.net_name)
        try:
            current_network.load(version='current')
        except ValueError:
            logger.fatal('Cannot evaluate a model without at least '
                         'a "current" version.')
            raise AssertionError('No current version of network.')

        nextgen_network = ChessNetwork(self.net_name)
        try:
            nextgen_network.load(version='nextgen')
        except ValueError:
            logger.warn('No nextgen version of this model - testing '
                        'play against blank slate version.')

        mcts_params = dotdict(
            n_sims=800,
            c_base=4.0,
            c_init=1.0,
            eps=0.155,
            resign_threshold=-0.85,
            temperature=0.90,
            use_noise=False
        )

        c_mcts = MCTS(current_network, params=mcts_params)
        ng_mcts = MCTS(nextgen_network, params=mcts_params)

        env = ChessEnvironment()

        agent_params = dotdict(
            temp_threshold=0,
            max_hmoves=100,
            n_book_moves=5
        )

        c_version = ChessAgent(c_mcts, env, params=agent_params)
        ng_version = ChessAgent(ng_mcts, env, params=agent_params)
        
        self.play_game(c_version, ng_version, env)

    def play_game(self, c_version, ng_version, env):

        step = 0
        while not env.is_game_over:
            step += 1
            use_book = step <= c_version.params.n_book_moves * 2
            if self.cur_white:
                if env.white_to_move:
                    c_version.move(step,
                                   use_book=use_book)
                else:
                    ng_version.move(step,
                                    use_book=use_book)
            else:
                if env.white_to_move:
                    ng_version.move(step,
                                    use_book=use_book)
                else:
                    c_version.move(step,
                                   use_book=use_book)
            if step % 2 == 0:
                logger.info(f'{step / 2} moves played in {self.game_name}')
            if step >= c_version.params.max_hmoves:
                env.adjudicate()

        winner = env.result_side()
        if self.cur_white and winner == 'White' or \
           (not self.cur_white and winner == 'Black'):
            winner = 'Current'
        elif (self.cur_white and winner == 'Black') or \
             (not self.cur_white and winner == 'White'):
            winner = 'Next-Gen'

        ocolor = 1 if not self.cur_white else -1
        logger.info(f'{self.game_name} over. Result: {env.result_string()} ({winner})')
        env.to_pgn(f'{c_version.search_tree.network.name} - Eval', self.game_name,
                   self_play_game=False, opponent_color=ocolor, opponent='Nextgen')
           


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--netname', dest='net_name', type=str,
                        help='The name of the network to use. A "current" '
                             'version of the network must exist.')
    args = parser.parse_args()
    net_name = args.net_name
    match_length = mp.cpu_count()

    color = False
    runners = []
    for i in range(match_length):
        color = not color
        runners.append(Runner.remote(net_name=net_name, iteration=i, cur_white=color))

    ray.get([r.run_evaluation.remote() for r in runners])

if __name__ == '__main__':
    main()
