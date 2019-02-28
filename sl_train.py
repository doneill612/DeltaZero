import argparse
import os
import io

import numpy as np
import chess.pgn as pgn

from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy, mean_squared_error
import keras.backend as K


from delta_zero.environment import ChessEnvironment
from delta_zero.network import ChessNetwork
from delta_zero.utils import labels
from delta_zero.logging import Logger

logger = Logger.get_logger('supervised-learning')

def train(net_name):
    net = ChessNetwork(net_name)
    try:
        net.load(version='current')
    except:
        pass
    gen = kingbase_generator(net.hparams.batch_size)
    with net.graph.as_default():
        with net.session.as_default():
            for e in range(net.hparams.epochs):
                logger.info(f'Epoch {e + 1}/{net.hparams.epochs}')
                for i, (X, y) in enumerate(gen):
                    # 20% validation split
                    if (i+1) % 5 == 0:
                        loss = net.model.test_on_batch(X, y)
                        if (i+1) % 30 == 0:
                            logger.info(f'val_loss: {loss[0]:.3f}, '
                                        f'val_pi_loss: {loss[1]:.3f}, '
                                        f'val_v_loss: {loss[2]:.3f}')
                    else:
                        loss = net.model.train_on_batch(X, y)
                        logger.info(f'loss: {loss[0]:.3f}, '
                                    f'pi_loss: {loss[1]:.3f}, '
                                    f'v_loss: {loss[2]:.3f}')
                    if (i+1) % 1000 == 0:
                        # save the model every 1000 steps (1 step = batch_size examples)
                        net.save(version='current')
                # save the model every epoch
                net.save(version='current')
                    
def clip(elo):
    y = (1. - np.exp(-.00122 * elo))
    return y

def kingbase_generator(batch_size):
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'delta_zero',
                             'data',
                             'supervised_learning', 'kingbase')
    env = ChessEnvironment()

    fn = os.path.join(directory, 'KingBaseLite2019-B00-B19.pgn')
    with open(fn, 'r', encoding="ISO-8859-1") as pgn_file:
        data = pgn_file.read()
        splits = iter(data.split('\n\n'))
        s_batch = np.empty(shape=(batch_size, 19, 8, 8))
        p_batch = np.empty(shape=(batch_size, 1968,))
        v_batch = np.empty(shape=(batch_size, 1))
        batch_idx = 0
        batches = 0
        for header, game in zip(splits, splits):
            sio = io.StringIO(f'{header}\n\n{game}')
            g = pgn.read_game(sio)
            elo1 = int(g.headers['WhiteElo'])
            elo2 = int(g.headers['BlackElo'])
            result = g.headers['Result'].split('-')[0]
            if result == '1/2':
                result = 0
            elif result == '1':
                result = 1
            else:
                result = 0
            moves = [m.uci() for m in list(g.mainline_moves())]
            n_moves = len(moves)
            for i in range(0, n_moves, batch_size):
                _moves = moves[i:min(i+batch_size, n_moves)]
                for i, m in enumerate(_moves):
                    # add board state to batch
                    s_batch[batch_idx] = env.canonical_board_state

                    # add policy vector to batch
                    policy = np.zeros(len(labels))
                    nonplayed = [x for x in env.legal_moves if x != m]
                    played_idx = np.where(labels==m)
                    nonplayed_idx = list(np.where(labels==n) for n in nonplayed)
                    clipped_p = clip(elo1 if (i % 2 == 0) else elo2)
                    try:
                        rem = (1. - clipped_p) / float(len(nonplayed))
                    except RuntimeWarning:
                        logger.warning('Divide by zero encountered, setting prior probs to 0.')
                        rem = 0
                    policy[played_idx] = clipped_p
                    np.put(policy, nonplayed_idx, rem)
                    p_batch[batch_idx] = policy

                    # add value to batch
                    v_batch[batch_idx] = result

                    env.push_action(m)
                    batch_idx += 1
                    # is the batch full? if so: yield, initialize new batch matricies
                    if batch_idx == batch_size:
                        batch_idx = 0
                        batches += 1
                        yield s_batch, [p_batch, v_batch]
                        s_batch = np.empty(shape=(batch_size, 19, 8, 8))
                        p_batch = np.empty(shape=(batch_size, 1968,))
                        v_batch = np.empty(shape=(batch_size, 1,))
            env.reset()
        # no more games, yield no matter what
        yield s_batch, [p_batch, v_batch]
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--netname', dest='net_name', type=str)

    args = parser.parse_args()
    net_name = args.net_name
    
    train(net_name)
