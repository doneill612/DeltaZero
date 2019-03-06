import sys
import os
import unittest

sys.path.append(os.path.join(os.path.pardir, os.path.dirname(__file__)))

from core.neuralnets.kerasnet import KerasNetwork

class KerasNetworkUnitTest(unittest.TestCase):
    
    def test_build(self):
        net = KerasNetwork('keras-net')

    def test_save_and_load(self):
        net = KerasNetwork('keras-net')
        net.save('current')
        net.save('nextgen')

        del net

        net = KerasNetwork('keras-net')
        net.load('current')
        net.load('nextgen')


if __name__ == '__main__':
    unittest.main()
