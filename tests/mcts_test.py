import sys
import os

sys.path.append(os.path.join(os.path.pardir, os.path.dirname(__file__)))
from core.oomcts import MCTS
from core.network import ChessNetwork
from core.environment import ChessEnvironment
from core.dzlogging import Logger

logger = Logger.get_logger('mcts-unittest')

if __name__ == '__main__':
    env = ChessEnvironment()
    net = ChessNetwork()
    for i in range(100):
        mcts = MCTS(net)
        while not env.is_game_over:
            pi = mcts.pi(env)
            action = pi['a']
            env.push_action(action)
            mcts.update(action)
