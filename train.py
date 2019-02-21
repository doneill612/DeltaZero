import argparse
import os

import numpy as np

from delta_zero.network import ChessNetwork
from delta_zero.logging import Logger

logger = Logger.get_logger('training')

def train(net_name, version):

    if version not in ('current', 'nextgen'):
        logger.fatal(f'Invalid version type: {version}. '
                     'Must be "nextgen" or "current"')
        raise AssertionError

    network = ChessNetwork(name=net_name)
    try:
        network.load(version=version)
    except ValueError

    train_examples = load_examples(net_name)
    network.train(train_examples)
    network.save(version=version)
        
    logger.info('Session complete')    
        

def load_examples(net_name):
    csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'delta_zero',
                           'data',
                           net_name,
                           'train')
    fn = os.path.join(csv_dir, 'train_set.npy')
    return np.load(fn)

    
if __name__ == '__main__':
    Logger.set_log_level('info')
    
    parser = argparse.ArgumentParser()

    parser.add_argument('net_name', type=str, help='Network name')
    parser.add_argument('version', type=str, type=int, help='Network version to save as - "current" or "nextgen"')
    
    args = parser.parse_args()
    net_name = args.net_name
    version = args.version
    train(net_name, version=version, warm_start=warm_start)
    
