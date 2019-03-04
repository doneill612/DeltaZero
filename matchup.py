from delta_zero.network import ChessNetwork
from delta_zero.mcts import MCTS
from delta_zero.agent import ChessAgent
from delta_zero.environment import ChessEnvironment
from delta_zero.dzlogging import Logger

logger = Logger.get_logger('matchup')
def game(net_name, ckpt=None):
    env = ChessEnvironment()

    w_net = ChessNetwork(net_name)
    b_net = ChessNetwork(net_name)
    
    w_net.load(version='nextgen', ckpt=ckpt)
    b_net.load(version='current')

    w_mcts = MCTS(w_net)
    b_mcts = MCTS(b_net)

    white = ChessAgent(w_mcts, env)
    black = ChessAgent(b_mcts, env)

    step = 0
    while not env.is_game_over:
        step += 1
        use_book = step <= white.params.n_book_moves * 2
        
        if env.white_to_move:
            white.move(step,
                       use_book=use_book)
            print(env)
        else:
            black.move(step,
                       use_book=use_book)
            print(env)
        
    winner = env.result_side()
    
    ocolor = -1
    logger.info(f'Game over. Result: {env.result_string()}')
    env.to_pgn(f'{white.search_tree.network.name} - Eval', 'selfplay',
               self_play_game=False, opponent_color=ocolor, opponent='Current')

    
if __name__ == '__main__':
    game('dzc', 60000)
