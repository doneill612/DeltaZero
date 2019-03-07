import argparse

from core.neuralnets.kerasnet import KerasNetwork
from core.dzlogging import Logger

logger = Logger.get_logger('supervised-learning')

def train(net_name, generator_tag, version='current', ckpt=None):
    
    net = KerasNetwork(net_name)
    generator = ExampleGenerator(generator_tag)
    
    if version not in ('current', 'nextgen'):
        logger.fatal(f'Invalid version type: {version}. '
                     'Must be "nextgen" or "current"')
        raise ValueError('Invalid version type')

    try:
        net.load(version=version, ckpt=ckpt)
    except RuntimeError as e:
        logger.warning('There was an error loading the model... '
                       'You are training a fresh model!')

    net.train_generator(generator)
                        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--netname', dest='net_name', type=str)
    parser.add_argument('--generator', dest='generator_tag', type=str)
    parser.add_argument('--checkpoint', nargs='?', default=None, dest='ckpt', type=int)
    parser.add_argument('--version', nargs='?', default='current', dest='version', type=str)


    args = parser.parse_args()
    net_name = args.net_name
    generator_tag = args.generator_tag
    version = args.version
    ckpt = args.ckpt
    
    train(net_name, generator_tag, version=version, ckpt=ckpt)
