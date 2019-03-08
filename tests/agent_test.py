import sys
import os
import unittest

sys.path.append(os.path.join(os.path.pardir, os.path.dirname(__file__)))

from core.neuralnets.kerasnet import KerasNetwork
from core.agent import ChessAgent
from core.environment import ChessEnvironment
from core.oomcts import MCTS

class AgentUnitTest(unittest.TestCase):

    def setUp(self):
        self.net = KerasNetwork('Dzc')
        self.net.load('current', 20001)
        self.env = ChessEnvironment()
        self.mcts = MCTS(self.net, noise=False, simulations=800)
        self.agent = ChessAgent(self.mcts, self.env)

    def test_playgame(self):
        self.agent.play('agent-test', show_moves=True)

if __name__ == '__main__':
    unittest.main()
