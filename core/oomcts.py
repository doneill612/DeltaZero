import numpy as np
from tqdm import tqdm

from .utils import labels

from .dzlogging import Logger

logger = Logger.get_logger('MCTS')

class Node(object):
    '''
    A node in the tree search as described in Silver et al.

    Each MCTS node stores a set of statistics regarding state-action pairs
    in the current environment.

    This implementation is an adjusted version of RocAlphaGo (repository link in README). 
    '''
    def __init__(self, parent, p):
        '''
        Constructs a new tree node.

        Params
        ------
            parent (Node) : The parent node of this tree node. If parent is None, this node is a root node.
            children (dict) : Child tree nodes belonging to this node. If dict has 0 length, this node is a leaf node.
                              The dict maps UCI notation chess moves to Node objects.
            n (int) : Visit count of the tree node
            q (float) : Tree node Q-value
            p (float) : Prior probability of the tree node
            u (float) : Visit count-adjusted prior probability of the tree node
        '''
        self.parent = parent
        self.children = {}
        self.p = p
        self.u = p
        self.q = 0
        self.n = 0

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_root(self):
        return self.parent is None

    def select(self):
        '''Selection function.

        Selects a child tree node with the highest Q+U value.

        Returns
        -------
           (action, Node) tuple that maximizes q + u
        '''
        return max(self.children.items(), key=lambda s_a_node: s_a_node[1].evaluate())
    
    def expand(self, action_p):
        '''Expansion function.

        Expands the search tree by populating child nodes according to a prior probability
        distribution returned by the policy network.

        Params
        ------
        action_p (ndarray(shape=(1968, 2))) : (action, prior probability) pairs returned by the policy network.
        '''
        for ap in action_p:
            action = ap[0]
            prior = ap[1]
            if action not in self.children:
                self.children[action] = Node(self, float(prior))

    def evaluate(self):
        '''Evaluation function.
        
        Calculates the adjusted Q value (q + u) for this tree node.

        Returns
        -------
            q + u for this node
        '''
        return self.q + self.u

    def update(self, v, c_puct):
        '''Update function.
        
        Runs a recursive update up the chain of tree nodes to the root node,
        updating tree node q and u values.
        
        Params
        ------
            v (float) : The q value returned by the value network
            c_puct (float) : The exploration parameter c (hyperparameter of the search tree).
        '''
        if self.parent:
            self.parent.update(v, c_puct)
        self._update(v, c_puct)

    def _update(self, v, c_puct):
        '''
        Increment the visit count for this tree node, adjust q and u values.

        Params
        ------
        v : float
            The q value returned by the value network
        c_puct : float
            The exploration parameter c (hyperparameter of the search tree).
        '''
        self.n += 1
        self.q += (v - self.q) / self.n
        if not self.is_root:
            self.u = c_puct * self.p * \
                     np.sqrt(self.parent.n) / (1 + self.n)

class MCTS(object):
    '''
    The neural network-driven Monte Carlo Search Tree.

    Actions are explored according to policy network out to specified depth (`d`). 
    Nodes in the tree at depth `d` are leaf nodes and are evaluated using a 
    value network. Values at each node in the tree updated via a backpropagation step.
    A single iteration to `d` is referred to as a "playout" - as many
    playouts are executed, the probability of visiting nodes in the search tree evolves.
    The action associated with the most visited node after each of the playouts is the selected
    move to make.

    The original AlphaGo architecture used separate policy and value networks, whereas AlphaZero
    (and DeltaZero) use a single residual network with policy and value heads. Therefore a single
    network is used to estimate a policy for tree node exploration and a value for tree node q-value updating.
    '''
    def __init__(self, network, c_puct=4., playout_depth=25, simulations=25):
        '''Constructs a Monte Carlo Search Tree.
        
        Params
        ------
            root (Node) : The root node (s_root in Silver et al.)
            network (core.network.NeuralNetwork) : The neural network with policy 
                                                   and value heads used for
                                                   policy and value estimation
            c_puct (float) : Exploration parameter
            playout_depth (int) : The node depth to reach when executing the playout phase
            simulations (int) : The number of playouts to perform
        '''
        self.root = Node(None, 1.)
        self.network = network
        self.c_puct = c_puct
        self.playout_depth = playout_depth
        self.simulations = simulations

    def playout(self, env):
        '''Performs a single playout.
        
        The root node is recursively expanded to a leaf node using the policy network
        to generate prior probabilities for possible actions to take. The value head 
        of the network is then used to estimate a q value at the leaf node, after which 
        the q value is backpropagated to the root node.

        NOTE : The `env` parameter is modified in place, so a copy is passed.

        Params
        ------
            env (core.environment.ChessEnvironment) : The current environment state 
                                                      from which to perform a playout.
        '''
        node = self.root
        for _ in range(self.playout_depth):
            if node.is_leaf:
                if env.is_game_over:
                    break
                action_p = self.network.predict(env.canonical_board_state)[0]
                action_p = self._mask_illegal(action_p, env)
                node.expand(np.column_stack((labels, action_p)))
            action, node = node.select()
            env.push_action(action)
            
        leaf_v = self.network.predict(env.canonical_board_state)[1]
        node.update(leaf_v, self.c_puct)

    def update(self, action):
        '''Update the tree search with a new root node.
        
        After an action is taken in the true environment, the root
        node (`s_root`, Silver et al.) is updated.

        Params
        ------
            action (str) : The action that was taken in the environment 
                           after the last simulation round.
        '''
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)

    def pi(self, env):
        '''Performs the full playout phase.
        
        After the playout phase an action is selected. The action
        corresponding to the tree node with the highest visit count is selected.

        Params
        ------
            env (core.environment.ChessEnvironment) : The current environment state 
                                                      from which to execute the full playout
                                                      phase.

        Returns
        -------
            res (dict) : A dictionary containing keys 'a', 'pr', and 'q', corresponding
                         to the chosen action, said action's prior probability, and expected
                         result (Q-value) respectively.
        '''
        for _ in range(self.simulations):
            self.playout(env.copy())

        res = {'a': None, 'pr': None, 'q': None}
        most_visited = max(self.root.children.items(),
                           key=lambda s_a_node: s_a_node[1].n)
        res['a'] = most_visited[0]
        res['pr'] = most_visited[1].p
        res['q'] = most_visited[1].q
        
        return res

    def _mask_illegal(self, action_p, env):
        '''Masks illegal moves with 0 probability in the given state.'''
        legal = env.legal_moves
        mask = np.isin(labels, legal, assume_unique=True).astype(np.int32)

        # mask illegal moves with 0 probability
        masked_action_p = action_p * mask
        # renormalize probability distribution
        s = np.sum(masked_action_p)
        if s > 0:
            masked_action_p = masked_action_p / s
        else:
            masked_action_p = masked_action_p + mask
            masked_action_p /= np.sum(masked_action_p)
            
        return masked_action_p
