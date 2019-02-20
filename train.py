import argparse

from delta_zero.network import ChessNetwork 

def train(net_name, warm_start=None):

    network = ChessNetwork(name=net_name)
    try:
        network.load(ckpt=str(warm_start))
    except ValueError as e:
        print(f'WARNING: {e}')


    train_examples = load_examples(net_name)
    network.train(train_examples)
    network.save(ckpt=str(s))
        
    print('Session complete')    
        

def load_examples(net_name):
    csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'delta_zero',
                           'data',
                           net_name,
                           'train')
    fn = os.path.join(csv_dir, 'train_set.npy')
    return np.load(fn)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('warm_start', nargs='?', type=int, help='Network version warm-start')
    
    args = parser.parse_args()
    warm_start = args.warm_start
    train( warm_start=warm_start)
    
