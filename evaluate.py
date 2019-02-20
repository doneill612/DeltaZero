from delta_zero.network import ChessNetwork 
from delta_zero.environment import ChessEnvironment
from delta_zero.agent import ChessAgent
from delta_zero.mcts import MCTS

def play_once():
    env = ChessEnvironment()

    network = ChessNetwork(name='delta_zero2')
    try:
        network.load(ckpt=str(4))
    except ValueError as e:
        print(f'WARNING: {e}')

    search_tree = MCTS(network)
    agent = ChessAgent(search_tree, env)

    agent.play(verbose=True)

if __name__ == '__main__':
    play_once()
