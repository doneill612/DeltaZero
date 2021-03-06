import numpy as np

from tqdm import tqdm

from .utils import dotdict, labels
from .dzlogging import Logger

EPS = 1e-8

def_params = dotdict(
    n_sims=25,
    c_puct=4.,
    c_base=4.0,
    c_init=1.0,
    eps=0.15,
    resign_threshold=None,
    temperature=1.,
    use_noise=True
)

logger = Logger.get_logger('MCTS')

class MCTS(object):

    def __init__(self, network, params=def_params):
        self.network = network
        self.params = params
        self.q_sa = {}
        self.n_sa = {}
        self.n_s = {}
        self.p_s = {}
        self.e_s = {}
        self.v_s = {}        

    def pi(self, env, temp, sims=None):
        
        if sims is None:
            sims = self.params.n_sims
        sim_vs = np.zeros(shape=(sims))

        for i in range(sims):
            sim_vs[i] = self._search(env.copy())

        v = np.max(sim_vs)

        res = {'a': None, 'pr': None, 'q': None}
        s = env.to_string()
        counts = [self.n_sa[(s, a_idx)] if (s, a_idx) in self.n_sa else 0
                  for a_idx in range(len(labels))]

        res['q'] = v
        
        if not temp:
            best_action_idx = np.argmax(counts)
            best_action = labels[best_action_idx]
            p = [0] * len(counts)
            p[best_action_idx] = 1
            res['a'] = best_action
            res['pr'] = np.asarray(p)
            return res
        else:
            counts = [c**(1. / self.params.temperature) for c in counts]
            p = [c / float(sum(counts)) for c in counts]
            if (self.params.resign_threshold and v > self.params.resign_threshold) \
               or not self.params.resign_threshold:
                res['a'] = np.random.choice(labels, p=p)
            res['pr'] = np.asarray(p)
            return res

    def reset(self):
        self.q_sa = {}
        self.n_sa = {}
        self.n_s = {}
        self.p_s = {}
        self.e_s = {}
        self.v_s = {}

    def _c_puct(self, s):
        c = np.log((1 + self.n_s[s] + self.params.c_base) / self.params.c_base) + self.params.c_init
        return c if c >= 4. else 4.

    def _alpha(self, env):
        n = len(env.legal_moves)
        if n <= 10:
            return 0.3
        if n > 10 and n <= 40:
            return 0.15
        if n > 40:
            return 0.03
        
    def _search(self, env):
        
        s = env.to_string()
        c_state = env.canonical_board_state
        
        if s not in self.e_s:
            self.e_s[s] = int(env.is_game_over)
        if self.e_s[s] != 0:
            return -self.e_s[s]

        if s not in self.p_s:
            self.p_s[s], v = self.network.predict(c_state)
            legal = np.asarray(env.legal_moves)
            legal_mask = np.isin(labels, legal, assume_unique=True)
            self.p_s[s] = self.p_s[s] * legal_mask
            
            sum_p_s_s = np.sum(self.p_s[s])
          
            if sum_p_s_s > 0:
                self.p_s[s] /= sum_p_s_s
            else:                
                self.p_s[s] = self.p_s[s] + legal_mask
                self.p_s[s] /= np.sum(self.p_s[s])

            self.v_s[s] = legal_mask
            self.n_s[s] = 0
            return -v

        valids = self.v_s[s]
        cur_best = -float('inf')
        best_action_idx = -1
        noise = np.random.dirichlet([self._alpha(env)] * len(labels))
        c = self._c_puct(s)
        for a_idx in range(len(labels)):
            if valids[a_idx]:
                p_ = self.p_s[s][a_idx]
                if self.params.use_noise:
                    p_ = (1. - self.params.eps) * p_ + self.params.eps * \
                         noise[a_idx]  
                if (s, a_idx) in self.q_sa:
                    u = self.q_sa[(s, a_idx)] + c * \
                        p_  * np.sqrt(self.n_s[s]) / \
                        (1 + self.n_sa[(s, a_idx)])
                else:
                    u = c *  p_ * np.sqrt(self.n_s[s] + EPS)
                if u > cur_best:
                    cur_best = u
                    best_action_idx = a_idx

        best_action = labels[best_action_idx]

        env.push_action(best_action)
        v = self._search(env)

        if (s, best_action_idx) in self.q_sa:
            self.q_sa[(s, best_action_idx)] = (
                self.n_sa[(s, best_action_idx)] *
                self.q_sa[(s, best_action_idx)] + v
            ) / (self.n_sa[(s, best_action_idx)] + 1)
            self.n_sa[(s, best_action_idx)] += 1
        else:
            self.q_sa[(s, best_action_idx)] = v
            self.n_sa[(s, best_action_idx)] = 1

        self.n_s[s] += 1
        return -v
        
