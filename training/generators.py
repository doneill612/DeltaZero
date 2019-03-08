import io
import os

import chess.pgn as pgn
import numpy as np

from engine import Stockfish
from core.environment import ChessEnvironment
from core.utils import labels
from core.dzlogging import Logger

logger = Logger.get_logger('ExampleGenerator')

GENERATORS = {
    'lichess': 'lichess_db_standard_rated_2013-07.pgn',
    'kingbase': 'KingBaseLite2019-C60-C99.pgn'
}

class ExampleGenerator(object):
    '''
    A wrapper class providing support for mini-batch training
    of the neural network. This class handles the batch preprocessing
    steps - namely reading the PGN files for supervised learning and
    generating training examples from the games.
    '''
    def __init__(self, tag):
        self.tag = tag
        self.floc = GENERATORS[self.tag]
        Stockfish.ENGINE.start()

    def get(self, batch_size):
        '''Gets the generator object which yields training examples.

        Training examples are tuples of the form,
        
            (`state`, `policy`, `value`)

        where `state`, `policy`, and `value` are ndarray objects with shape,
            
            state: (batch_size, 19, 8, 8)
            policy: (batch_size, 1968)
            value: (batch_size, 1)

        Params
        ------
            batch_size : the batch size to use when yielding examples

        Returns
        -------
            a generator that yields training examples sequentially
        '''
        directory = os.path.join(os.path.pardir,
                                 'data',
                                 'supervised_learning')
        fn = os.path.join(directory, self.floc)
        env = ChessEnvironment()
        with open(fn, 'r', encoding="ISO-8859-1") as pgn_file:
            data = pgn_file.read()
            splits = iter(data.split('\n\n'))
            states, policies, values = self._empty_batch(batch_size)
            batch_idx = 0; games = 0; batches = 0;
            for header, moves in zip(splits, splits):
                sio = io.StringIO(f'{header}\n\n{moves}')
                game = pgn.read_game(sio)
                try:
                    white_elo = int(game.headers['WhiteElo'])
                    black_elo = int(game.headers['BlackElo'])
                except:
                    continue
                if white_elo < 1700 or black_elo < 1700:
                    continue
                moves = [m.uci() for m in list(game.mainline_moves())]
                n_moves = len(moves)
                for i in range(0, n_moves, batch_size):
                    move_batch = moves[i:min(i+batch_size, n_moves)]
                    for m in move_batch:
                        states[batch_idx] = env.canonical_board_state
                        policies[batch_idx] = self._build_policy(env, m, white_elo, black_elo)
                        values[batch_idx] = Stockfish.ENGINE.evaluate(env)
                        env.push_action(m)
                        batch_idx += 1
                        if batch_idx == batch_size:
                            batch_idx = 0
                            batches += 1
                            yield states, [policies, values]
                            states, policies, values = self._empty_batch(batch_size)
                env.reset()
                games += 1
            yield states, [policies, values]
            env.reset()
            Stockfish.ENGINE.close()
                

    def _build_policy(self, env, m, white_elo, black_elo):
        policy = np.zeros(len(labels))
        nonplayed = [x for x in env.legal_moves if x != m]
        played_idx = np.where(labels==m)
        nonplayed_idx = list(np.where(labels==n) for n in nonplayed)
        clipped_p = 0
        if env.white_to_move:
            clipped_p = self._clip(white_elo)
        else:
            clipped_p = self._clip(black_elo)

        n_nonplayed = len(nonplayed)
        if n_nonplayed > 0:
            rem = (1. - clipped_p) / float(n_nonplayed)
            policy[played_idx] = clipped_p
            np.put(policy, nonplayed_idx, rem)
        else:
            policy[played_idx] = 1.
        
        return policy


    def _clip(self, elo):
        return (1. - np.exp(-.000822 * elo))
    
    def _empty_batch(self, batch_size):
        states = np.empty(shape=(batch_size, 19, 8, 8))
        policies = np.empty(shape=(batch_size, 1968))
        values = np.empty(shape=(batch_size, 1))
        return states, policies, values
    
