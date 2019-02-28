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

#### Network architecture

The network architecture mimics closely the design of AlphaZero. DeltaZero uses a convolutional res-net, or residual network. The network consists of a body followed by two heads - a policy head and a value head. The policy represents move probabilities given a particular board state, and the value represents the estimated outcome of the game, [-1.0, 1.0], with -1.0 being a loss, 0.0 being a draw, and 1.0 being a win.

The network body consists of a linear rectified, batch-normalized convolutional layer with 64 filters, kernel size `3x3` and stride 1, followed by (currently) 19 residual layers (note that the number of residual layers is currently evolving through different model iterations). Each residual layer consists of two skip-connected, linear rectified, batch-normalized convolutional layers of kernel size `3x3` and stride 1. Both convolutional layers in each residual layer also have 64 filters.

The policy head is slightly different from the one implemented in AlphaZero, which is due to the fact that the policy is represented differently in DeltaZero. AlphaZero represents the chess policy as a `73x8x8` matrix, while DeltaZero represents its policy as a flat vector of length `1968`, each element representing a possible move on the chess board. The policy head takes the output of the residual layers and passes them through a single linear rectified, batch-normalized convolutional layer with 32 filters, kernel size `1x1` and stride 1. This convoluted output is then flattened and passed through a linear rectified fully-connected layer of size 2048. This dense layer leads to a final softmax layer of size 1968 - the output of which represents the policy estimate.

The value head is also fed by the output of the residual layers, and consists of a single linear rectified, batch-normalized convolutional layer with a single filter, kernel size `1x1` and stride 1. The convoluted output is flattened and passed through a linear rectified fully-connected layer of size 256 with 40% dropout, followed by a `tanh` layer - the output of which represents the value estimate.

#### Supervised learning warm-start

In the interest of saving years of computation time (or thousands of dollars on distributed computing), the network is "warm-started" with supervised training data from a database containing roughly 1 million chess games in PGN format. At the time of writing, the current iteration of DeltaZero has been trained on approximately 56,000 games.

For each game, a single training example is extracted in the following manner:

- The game is loaded into a `ChessEnvironment` object.
- At each move, the board state is encoded into a `(19, 8, 8)` matrix.
- The policy for the move is created in the following manner:
  - The ELO ratings of each player are recorded. The move that was actually played in the game is assigned a high probability, proportional to the ELO rating of the player that made the move. The probablity is calculated as `P = 1 - exp(a*e)`, where `e` is the ELO rating and `a = -0.00122` (found by trial and error). The remainder probability (`1 - P`) is divided amongst the rest of the legal moves in the position. The illegal moves are masked with a probability of 0.
  - For example, a player with an ELO rating of 2000 (minumum ELO of any player in the training games) would have a probability of `0.913` assigned to the move he/she played in a game, while a player with an ELO rating of 2750 would have a probability of `0.965` assigned to a move.
- The value is the result of the game - 1 for a white win, 0 for a draw, and -1 for a black win.

One training example per move is generated. DeltaZero trains with a batch size of 256, and uses `Adam` as an optimizer, with an initial learning rate of 0.001.

### mcts.py

The `mcts` module contains a class definition `MCTS` which is an implementation of the Monte Carlo Tree Search algorithm, slightly modified to use the predictions of the current iteration of the neural network to generate move probabilities. The MCTS hyperparameters are almost identical to those specified in the AlphaZero paper.

When playing a game of chess, the `ChessAgent` in order to make a move will run $N$ simulations of MCTS in order to maximize the estimated value output of the game. Higher `N` values can be equated to deeper thinking. During evaluation games, `N = 800` is chosen. Different values of $N$ are used to generate more varied games, and to gauge the prediction strength of the current network.