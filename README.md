# DeltaZero

`DeltaZero` is a python implementation of the AlphaZero architecture for chess. Design ideas were
drawn from a couple of existing open-source python implementations, namely
[alpha-zero-general](https://github.com/suragnair/alpha-zero-general) and
[chess-alpha-zero](https://github.com/Zeta36/chess-alpha-zero). My goal is to learn from these implementations,
and make readability/design improvements along the way.

## Actors

### `environment.py`

The `environment` module contains a class declaration for the `ChessEnvironment`, which is repsonsible
for managing information about the state of the chess board, and translating that information into
a format interpretable by a convolutional neural network.

#### Board state representation

DeltaZero uses [`python-chess`](https://github.com/niklasf/python-chess) to do a fundamental board
representation. `python-chess` is an extremely well-written chess library completely implemented in Python.
It is well documented, and the source code is very readable and understandable.

The state of a chess game is encoded into a `(19, 8, 8)` numpy array, with the first dimension representing
what are referred to as "planes" (or channels), and the second two dimensions representing the shape of the
chess board (8 by 8 squares). The first 12 planes are bit matricies representing the positions of the white
and black pieces. For example, the plane at index `0` represents the white pawn positions, and at the start
of the game would look like this:

```
[
 0, 0, 0, 0, 0, 0, 0, 0
 0, 0, 0, 0, 0, 0, 0, 0
 0, 0, 0, 0, 0, 0, 0, 0
 0, 0, 0, 0, 0, 0, 0, 0
 0, 0, 0, 0, 0, 0, 0, 0
 0, 0, 0, 0, 0, 0, 0, 0
 1, 1, 1, 1, 1, 1, 1, 1
 0, 0, 0, 0, 0, 0, 0, 0
]
```

The last 7 planes contain game metadata - castling rights for both sides, the *en passant* square, fifty-move counter,
and side-to-move.

#### Adjudication

The `ChessEnvironment` can also adjudicate games, bringing an immediate end to a game in progress. This is helpful
in the early training stages when the Agent makes essentially random moves, and the games last forever. Adjudication
is performed by using Stockfish 10, one of the world's strongest open-source chess engines. At the time of adjudication,
a position evaluation is performed, and the game result is determined based on this evaluation.

#### PGN

Games played by DeltaZero can be exported to PGN format for review.

### `agent.py`

The `agent` module contains a class declaration for the `ChessAgent`, which is the entity that traverses the `ChessEnvironment` by
taking actions (making chess moves) and receiving rewards. The `ChessAgent` executes episodes of self-play during which it plays
chess against itself, generating training examples in the form of board positions + corresponding game outcome.

#### *Aside regarding computational resources*

DeltaZero implements a multiprocessing module called `selfplay.py` dedicated to parallelizing the self-play process. During self-play,
a neural network-based Monte Carlo Tree Search is performed to generate actions for the Agent to take. Although the algorithm itself
is computationally efficient, a very large number of training examples (and therefore a large number of self-play games) are needed.
Realistically speaking, without access to an extremely high-power distributed environment (see[Lc0](https://github.com/LeelaChessZero/lc0)), generating self-play games simply isn't feasible to bring the network to superhuman playing levels. Training methods are discussed in the coming sections.

#### Opening book

The `ChessAgent` also has optional access to an opening book for evaluation sessions against either a previous version, or human/engine opponents.

### `network.py`

The `network` module contains two class declarations: 1) An abstract base class `NeuralNetwork`, containing abstract methods for building the network architecture, training, running predictions, and saving/loading weights. 2) A Keras-based implementation of `NeuralNetwork` called `ChessNetwork`.