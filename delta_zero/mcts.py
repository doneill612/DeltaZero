import numpy as np

from .utils import dotdict, labels
from .logging import Logger

EPS = 1e-8

def_params = dotdict(
    n_sims=5,
    cpuct=1.0,
    alpha=0.3,
    eps=0.25,
    resign_threshold=-0.8,
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

    def pi(self, env, temp):

        sim_vs = []
        for i in range(self.params.n_sims):
            sim_vs.append(self._search(env))

        v = np.max(sim_vs)

        res = {'a': None, 'pr': None, 'v': None}
        s = env.to_string()
        counts = [self.n_sa[(s, a_idx)] if (s, a_idx) in self.n_sa else 0
                  for a_idx in range(len(labels))]

        res['v'] = v
        
        if temp == 0:
            best_action_idx = np.argmax(counts)
            best_action = labels[best_action_idx]
            p = [0] * len(counts)
            p[best_action_idx] = 1
            res['a'] = best_action
            res['pr'] = np.asarray(p)
            return res
        else:
            counts = [c**(1. / temp) for c in counts]
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
        
    def _search(self, env):

        env = env.copy()
        s = env.to_string()
        c_state = env.canonical_board_state
        
        if s not in self.e_s:
            self.e_s[s] = env.is_game_over
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
        noise = np.random.dirichlet([self.params.alpha] * len(labels))

        for a_idx in range(len(labels)):
            if valids[a_idx]:
                p_ = self.p_s[s][a_idx]
                p_ = (1. - self.params.eps) * p_ + self.params.eps * \
                     noise[a_idx]  
                if (s, a_idx) in self.q_sa:
                    u = self.q_sa[(s, a_idx)] + self.params.cpuct * \
                        p_  * np.sqrt(self.n_s[s]) / \
                        (1 + self.n_sa[(s, a_idx)])
                else:
                    u = self.params.cpuct * self.p_s[s][a_idx] * \
                        np.sqrt(self.n_s[s] + EPS)
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
        
                         
            

        
        
