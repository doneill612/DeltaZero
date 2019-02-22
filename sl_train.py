import argparse
import os
import io

import numpy as np
import chess.pgn as pgn

from tqdm import tqdm

from delta_zero.environment import ChessEnvironment
from delta_zero.network import ChessNetwork
from delta_zero.utils import labels
from delta_zero.logging import Logger

logger = Logger.get_logger('supervised-learning')

def train(net_name):
    
    network = ChessNetwork(net_name)
    state, target_pi, target_v = load_sl_examples()

    # manually recreate network.train(), as the examples
    # do not need to undergo reformatting
    with network.graph.as_default():
        with network.session.as_default():
            network.model.fit(state, [target_pi, target_v],
                              batch_size=network.hparams.batch_size,
                              epochs=network.hparams.epochs)

    network.save(version='current')

def load_sl_examples():
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'delta_zero',
                             'data',
                             'supervised_learning')
    fn = os.path.join(directory, 'Kasparov.pgn')
    game = pgn.read_game(open(fn))

    env = ChessEnvironment()
    exs = []
    with open(fn, 'r', encoding="ISO-8859-1") as pgn_file:
        data = pgn_file.read()
        splits = data.split('\n\n')
        for i in tqdm(range(0, len(splits), 2)):
            sio = io.StringIO(f'{splits[i]}\n\n{splits[i+1]}')
            exs.append(extract_game_examples(pgn.read_game(sio), env))

    state, target_pi, target_v = list(zip(*exs))

    state = np.concatenate([s for s in state])
    target_pi = np.concatenate([p for p in target_pi])
    target_v = np.concatenate([v for v in target_v])

    logger.info(f'Extracted {state.shape[0]} examples.')

    return state, target_pi, target_v
    
def extract_game_examples(game, env):
    env.reset()
    states = []
    tar_policies = []
    tar_values = []
    for move in game.mainline_moves():
        env.push_action(move.uci())
    
        states.append(env.canonical_board_state)
        
        mask = np.isin(labels, move.uci(), assume_unique=True).astype(np.int32)
        tar_p = np.ones(len(labels), dtype=np.int32) * mask
        tar_policies.append(tar_p)
        
        res = game.headers['Result']
        if res.split('-')[0] == '1/2':
            v = 0
        elif res.split('-')[0] == '1':
            v = 1
        else:
            v = -1
        tar_values.append(v) 

        
    states = np.array([s for s in states])
    tar_policies = np.array([p for p in tar_policies])
    tar_values = np.array([v for v in tar_values])
    return states, tar_policies, tar_values

    
if __name__ == '__main__':
    
    Logger.set_log_level('info')
    parser = argparse.ArgumentParser()

    parser.add_argument('net_name', type=str)

    args = parser.parse_args()
    net_name = args.net_name
    
    train(net_name)
