import argparse
import os
import io
import time

import numpy as np
import chess.pgn as pgn

from core.environment import ChessEnvironment
from core.network import ChessNetwork
from core.utils import labels
from core.dzlogging import Logger

logger = Logger.get_logger('supervised-learning')

EPS = 1e-6

def train(net_name, version='current', ckpt=None):
    net = ChessNetwork(net_name)

    if version not in ('current', 'nextgen'):
        logger.fatal(f'Invalid version type: {version}. '
                     'Must be "nextgen" or "current"')
        raise ValueError('Invalid version type')

    try:
        net.load(version=version, ckpt=ckpt)
    except:
        pass


    start = time.time()
    for e in range(net.hparams.epochs):
        gen = lichess_generator(net.hparams.batch_size)
        logger.info(f'Epoch {e + 1}/{net.hparams.epochs}')
        for i, (X, y) in enumerate(gen):
            loss = net.batch_train(X=X, y=y)
            if (i+1) % 10 == 0:
                logger.info(f'loss: {loss[0]:.3f}, '
                        f'pi_loss: {loss[1]:.3f}, '
                        f'v_loss: {loss[2]:.3f}')
                elapsed = time.time() - start
                elapsed_str = f'{elapsed:.3f}'
                mins = elapsed // 60
                if mins >= 1:
                    elapsed_str = f'{int(mins)} min. {elapsed - 60 * mins:.3f} sec.'
                logger.info(f'Elapsed: {elapsed_str} sec.')
                if (i+1) % 100 == 0:
                    net.flush_writer(verbose=True)
            if (i+1) % 10000 == 0:
                # save the model every 1000 steps (1 step = batch_size examples)
                net.save(version=version, ckpt=(i+1))
        # save the model every epoch
        net.save(version=version)
                    
def clip(elo):
    y = (1. - np.exp(-.000822 * elo))
    return y

def lichess_generator(batch_size):
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'delta_zero',
                             'data',
                             'supervised_learning')
    env = ChessEnvironment()

    fn = os.path.join(directory, 'lichess_db_standard_rated_2013-07.pgn')
    with open(fn, 'r', encoding="ISO-8859-1") as pgn_file:
        data = pgn_file.read()
        splits = iter(data.split('\n\n'))
        s_batch = np.empty(shape=(batch_size, 19, 8, 8))
        p_batch = np.empty(shape=(batch_size, 1968,))
        v_batch = np.empty(shape=(batch_size, 1))
        batch_idx = 0
        games = 0
        batches = 0
        for header, game in zip(splits, splits):
            sio = io.StringIO(f'{header}\n\n{game}')
            g = pgn.read_game(sio)
            try:
                elo1 = int(g.headers['WhiteElo'])
                elo2 = int(g.headers['BlackElo'])
            except:
                continue
            if elo1 < 1700 or elo2 < 1700:
                continue
            result = g.headers['Result'].split('-')[0]
            if result == '1/2':
                result = 0
            elif result == '1':
                result = 1
            else:
                result = -1
            moves = [m.uci() for m in list(g.mainline_moves())]
            n_moves = len(moves)
            for i in range(0, n_moves, batch_size):
                _moves = moves[i:min(i+batch_size, n_moves)]
                for i, m in enumerate(_moves):
                    # add board state to batch
                    s_batch[batch_idx] = env.canonical_board_state

                    # add policy vector to batch
                    policy = np.zeros(len(labels))
                    legals = env.legal_moves
                    nonplayed = legals[legals != m]
                    played_idx = np.where(labels==m)
                    nonplayed_idx = list(np.where(labels==n) for n in nonplayed)
                    clipped_p = 0
                    if env.white_to_move:
                        clipped_p = clip(elo1)
                    else:
                        clipped_p = clip(elo2)
                    n_nonplayed = len(nonplayed)
                    if n_nonplayed > 0:
                        rem = (1. - clipped_p) / float(n_nonplayed)
                    else:
                        clipped_p = 1. - EPS
                        rem = EPS
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
            games += 1
            if games % 10 == 0:
                logger.info(f'Processed {games} games, {batches} steps ({batches * batch_size} positions)') 
        # no more games, yield no matter what
        yield s_batch, [p_batch, v_batch]
        env.reset()

def kingbase_generator(batch_size):
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'delta_zero',
                             'data',
                             'supervised_learning', 'kingbase')
    env = ChessEnvironment()

    fn = os.path.join(directory, 'KingBaseLite2019-C60-C99.pgn')
    with open(fn, 'r', encoding="ISO-8859-1") as pgn_file:
        data = pgn_file.read()
        splits = iter(data.split('\n\n'))
        s_batch = np.empty(shape=(batch_size, 19, 8, 8))
        p_batch = np.empty(shape=(batch_size, 1968,))
        v_batch = np.empty(shape=(batch_size, 1))
        batch_idx = 0
        games = 0
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
                result = -1
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
                    clipped_p = 0
                    if env.white_to_move:
                        clipped_p = clip(elo1)
                    else:
                        clipped_p = clip(elo2)

                    n_nonplayed = len(nonplayed)
                    if n_nonplayed > 0:
                        rem = (1. - clipped_p) / float(n_nonplayed)
                    else:
                        clipped_p = 1. - EPS
                        rem = EPS
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
            games += 1
            if games % 10 == 0:
                logger.info(f'Processed {games} games, {batches} steps ({batches * batch_size} positions)') 
        # no more games, yield no matter what
        yield s_batch, [p_batch, v_batch]
        env.reset()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--netname', dest='net_name', type=str)
    parser.add_argument('--checkpoint', nargs='?', default=None, dest='ckpt', type=int)
    parser.add_argument('--version', nargs='?', default='current', dest='version', type=str)


    args = parser.parse_args()
    net_name = args.net_name
    version = args.version
    ckpt = args.ckpt
    
    train(net_name, version=version, ckpt=ckpt)
