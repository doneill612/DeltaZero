import copy
import os

import chess
import numpy as np

import chess.uci as uci

PIECES = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(1, 7)
COLORS = [WHITE, BLACK] = [True, False]

RESULT = {
    0  : {'side': 'Draw', 'res': '1/2-1/2', 'key': 0},
    1  : {'side': 'White', 'res': '1-0', 'key': 1},
    -1 : {'side': 'Black', 'res': '0-1', 'key': -1}
}


class ChessEnvironment(object):
    '''Represents the chess board.

    This class is backed by the `chess` library (https://github.com/niklasf/python-chess).
    The chess library uses bit boards as internal representation, and this class uses
    numpy arrays.

    The state of a chess board in the context of a reinforcement learning application
    is represented as a (18, 8, 8) tensor - qualitatively, 18 'planes' of 8x8 boards, the
    values of each plane representing information about the game in progress.

        - planes 1-6 => white piece positions
        - planes 7-12 => black piece positions
        - planes 13-18 => castling rights, fifty-move rule, en-passant square

    Actions (chess moves) are represented in UCI notation, and are pushed to the board's
    move stack.
    '''

    
    def __init__(self, board=chess.Board()):
        '''
        Initializes the chess environment.

        Args
        ----
            board: a `chess.Board` object
        '''
        self.board = board
        self.winner = None

    @property
    def legal_moves(self):
        '''
        Returns a list of the current legal moves in the position.
        The moves are represented in UCI notation.

        Returns
        -------
            list: a list of legal moves in uci notation
        '''
        return list(m.uci() for m in self.board.legal_moves)

    @property
    def white_to_move(self):
        '''
        Checks to see if it is white's turn.

        Returns
        -------
            bool: True if it is white to move, False otherwise
        '''
        return self.board.turn

    @property
    def fen(self):
        return self.board.fen()

    @property
    def canonical_board_state(self):
        '''
        Returns the board state from the POV of the current side
        to play (WHITE or BLACK). 

        This equates to a 180 degree  rotation of each of the planes 
        in the board state if the side to move is BLACK.

        Returns
        -------
            np.ndarray: The board state from the POV of the current side to play.
        '''
        state = self.board_state
        if not self.white_to_move:
            for i, plane in enumerate(state):
                state[i] = np.rot90(plane, k=2)
        return state

    @property
    def board_state(self):
        '''
        Stacks the piece planes and auxiliary planes into a single
        (18, 8, 8) array. 

        This (18, 8, 8) array represents the chess board state from white's POV.

        Returns
        -------
            np.ndarray: The board state from white's POV
        '''
        piece = self.build_piece_planes()
        auxiliary = self.build_auxiliary_planes()
        return np.vstack([piece, auxiliary])
    
    @property
    def is_game_over(self):
        '''
        Checks whether or not the board state constitutes a game that has ended.

        Returns
        -------
            bool: True if the game is over, False otherwise
        '''
        res = self.board.result(claim_draw=True) # claim_draw=True => Check 50 move rule
        if res != '*':
            if res == '1-0':
                self._end_game(1)
            elif res == '0-1':
                self._end_game(-1)
            else:
                self._end_game(0)
        return self.winner != None

    def adjudicate(self):

        handler = uci.InfoHandler()
        ep = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'stockfish-10-win',
                          'Windows',
                          'stockfish_10_x64.exe')
        engine = uci.popen_engine(ep)
        engine.info_handlers.append(handler)
        engine.position(self.board)
        evaltime = 1000
        evalu = engine.go(movetime=evaltime)
        score = handler.info['score'][1].cp
        if score is None:
            score = f'Mate in {handler.info["score"][1].mate}'
        else:
            score = score / 100.

        if isinstance(score, str):
            self._end_game(1 if self.white_to_move else -1)
        else:
            if score > 2.:
                self._end_game(1)
            elif score < -2:
                self._end_game(-1)
            else:
                self._end_game(0)

    def push_action(self, action_uci, verbose=True):
        '''
        Updates the internal board representation by pushing an "action" to the move stack.

        Params
        ------
            action_uci (str): a string representing the move to make in UCI notation

        '''
        if action_uci is not None:
            self.board.push_uci(action_uci)
            
        else:
            if verbose:
                print(f'{"White" if self.white_to_move else "Black"} resigns.')
            self._send_resignation()

    
    def _send_resignation(self):
        '''
        Forcefully transitions the environment into a finished game state.

        The winner becomes the opposite of the side to move.
        '''
        self._end_game(-1 if self.white_to_move else 0)

    def reset(self):
        '''
        Resets the state of the environment.
        '''
        self.winner = None
        self.board = chess.Board()

    def copy(self):
        return copy.deepcopy(self)

    def to_string(self):
        return self.canonical_board_state.tostring()
        
    def _end_game(self, winner):
        self.winner = RESULT[winner]

    def result_side(self):
        '''
        Returns a string representation of the side who won the game.

        Returns
        -------
            str: "White", "Black", or "Draw"
        '''
        if self.winner:
            return self.winner['side']

    def result_string(self):
        '''
        Returns a formatted string representing the result of the game.

        Returns
        -------
            str: "1-0" (white wins), "0-1" (black wins), or "1/2-1/2" (draw)
        '''
        if self.winner:
            return self.winner['res']

    def result_value(self):
        if self.winner:
            return self.winner['key']

    def build_piece_planes(self):
        '''
        Builds a (12, 8, 8) bit-array representing the positions
        of each piece (black and white PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING).

        Returns
        -------
            np.ndarray: A group of planes representing the positions of every piece
                        on the board
        '''
        p_planes = np.zeros(shape=(12, 8, 8), dtype=np.int32)
        pidx = 0
        for color in COLORS:
            for piece in PIECES:
                idxs = []
                for flat_idx in self.get_pieces(piece, color):
                    idx = np.unravel_index(flat_idx, dims=(8,8))
                    idxs.append(idx)
                for r, f in idxs:
                    p_planes[pidx][r][f] = 1
                pidx += 1
        return np.flip(p_planes, axis=1)
        
    def build_auxiliary_planes(self):
        '''
        Builds a (6, 8, 8) representation of auxiliary
        information (castling rights, en-passant square, fifty-move rule)
        about the current chess board.
        '''
        [KSC_WHITE, QSC_WHITE, KSC_BLACK, QSC_BLACK, FIFTY_MOVE, EPS] = range(6)
        
        auxiliary_planes = np.zeros(shape=(6, 8, 8), dtype=np.int32)
        fen = self.board.fen()
        splits = fen.split(' ')

        castling_str = splits[2]
        auxiliary_planes[KSC_WHITE] = int('K' in castling_str)
        auxiliary_planes[QSC_WHITE] = int('Q' in castling_str)
        auxiliary_planes[KSC_BLACK] = int('k' in castling_str)
        auxiliary_planes[QSC_BLACK] = int('q' in castling_str)

        auxiliary_planes[FIFTY_MOVE] = int(splits[4])
        
        en_passant_sq = splits[3]
        if en_passant_sq != '-':
            r, f = self.sq_to_coord(en_passant_sq)
            auxiliary_planes[EPS][r][f] = 1
                                  
        return auxiliary_planes

    def get_pieces(self, ptype, pcolor):
        '''
        Gets the flat indicies (square numbers) of a particular piece of a particular color
        on the current board.

        Params
        ------
            ptype (int): the piece type
            pcolor (bool): the piece color

        Example
        -------
            >>> env = ChessEnvironment()
            >>> env.push_action('d2d4')
            >>> print(env.get_pieces(PAWN, WHITE))
            [8, 9, 10, 11, 13, 14, 15, 28]

        Returns
        -------
            list: Flat indicies of the requested piece type and color
        '''
        return list(self.board.pieces(ptype, pcolor))

    def sq_to_coord(self, sq):
        r = 8 - int(sq[1])
        f = ord(sq[0]) - ord('a')
        return r, f

    def coord_to_sq(self, coord):
        return f"{chr(ord('a') + coord[1])}{str(8 - coord[0])}"

    def __repr__(self):
        return str(self.board)
            

if __name__ == '__main__':

    env = ChessEnvironment()
    env.push_action('g2g4')
    env.push_action('e7e6')
    env.adjudicate()
    print(env.winner)
    
