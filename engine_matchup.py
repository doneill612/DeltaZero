import argparse
import os
import platform

import chess.engine as eng

from delta_zero.agent import ChessAgent
from delta_zero.environment import ChessEnvironment
from delta_zero.mcts import MCTS
from delta_zero.network import ChessNetwork
from delta_zero.dzlogging import Logger
from delta_zero.utils import dotdict

logger = Logger.get_logger('engine-matchup')

mcts_params = dotdict(
    n_sims=1000,
    c_base=4.0,
    c_init=1.0,
    eps=0.155,
    resign_threshold=-0.85,
    temperature=1.,
    use_noise=False
)

engine_params = dotdict(
    depth=2,
    time=0.01,
    nodes=5
)

def matchup(net_name, play_black=False):

    env = ChessEnvironment()
    
    network = ChessNetwork(name=net_name)
    network.load(version='current')

    mcts = MCTS(network, params=mcts_params)
    agent = ChessAgent(mcts, env)

    current_os = platform.system()
    ep = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delta_zero', 'data',
                      f'stockfish-10-{"win" if current_os == "Windows" else "linux"}',
                      f'{current_os}',
                      f'stockfish_10_x64{".exe" if current_os == "Windows" else ""}')


    engine = eng.SimpleEngine.popen_uci(ep)
    logger.info('Engine loaded')

    play_game(agent, engine, play_black)

def play_game(agent, engine, play_black):

    step = 1
    while not agent.env.is_game_over:
        use_book = step <= agent.params.n_book_moves * 2
        if (agent.env.white_to_move and not play_black) \
           or (not agent.env.white_to_move and play_black):
            agent.move(step, use_book=use_book)        
        else:
            einfo = engine.play(agent.env.board,
                                eng.Limit(time=engine_params.time,
                                          depth=engine_params.depth,
                                          nodes=engine_params.nodes))
            emove = einfo.move.uci()
            agent.env.push_action(emove)

        if step % 2 == 0:
            logger.info(f'{step / 2} moves played')
            print(f'{agent.env}\n')

        step += 1
        

    ocolor = 1 if play_black else -1
    agent.env.to_pgn(agent.search_tree.network.name, 'dummy stockfish matchup', opponent='dummy stockfish',
                     opponent_color=ocolor)
    engine.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--netname', dest='net_name', type=str)
    parser.add_argument('--playblack', action='store_true', dest='play_black')

    args = parser.parse_args()
    
    net_name = args.net_name
    play_black = args.play_black

    matchup(net_name, play_black=play_black)

