import sys
import os
import unittest

sys.path.append(os.path.join(os.path.pardir, os.path.dirname(__file__)))

from training.generators import ExampleGenerator
from core.dzlogging import Logger

logger = Logger.get_logger('unit-test')

class ExampleGeneratorUnitTest(unittest.TestCase):

    def setUp(self):
        self.gen = ExampleGenerator('testgen', 'KingBaseLite2019-C60-C99.pgn')
        self.batch_size = 64

    def test_get_gen(self):
        for i, exs in enumerate(self.gen.get(batch_size=self.batch_size)):
            logger.info(f'Batch {i} delivered, shapes: {exs[0].shape}, {exs[1][0].shape}, {exs[1][1].shape}')
            if i == self.batch_size:
                break

if __name__ == '__main__':
    unittest.main()
