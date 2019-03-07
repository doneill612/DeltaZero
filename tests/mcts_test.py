import sys
import os
import unittest

sys.path.append(os.path.join(os.path.pardir, os.path.dirname(__file__)))
from core.oomcts import MCTS
from core.neuralnets.kerasnet import KerasNetwork
from core.environment import ChessEnvironment
from core.dzlogging import Logger

logger = Logger.get_logger('mcts-unittest')

class MCTSUnitTest(unittest.TestCase):

    def setUp(self):
        self.env = ChessEnvironment()
        self.net = KerasNetwork()
        self.mcts = MCTS(self.net)

    def test_gameplay(self):
        while not self.env.is_game_over:
            pi = self.mcts.pi(self.env)
            action = pi['a']
            self.env.push_action(action)
            self.mcts.update(action)
            print(self.env)

if __name__ == '__main__':
    unittest.main()
