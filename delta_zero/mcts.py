import numpy as np

from utils import dotdict, labels

EPS = 1e-8

def_params = dotdict(
    n_sims=10
)

class MCTS(object):

    def __init__(self, agent, network, params):
        self.env = agent.env.copy()
        self.network = network
        self.params = params
        self.q_sa = {}
        self.n_sa = {}
        self.n_s = {}
        self.p_s = {}
        self.e_s = {}
        self.v_s = {}

    def pi(self, c_state, temp=1):

        v = self._search(self.env)

        res = {'a': None, 'pr': None}
        s = self.env.to_string()
        counts = [self.n_sa[(s, a_idx)] if (s, a_idx) in self.n_sa else 0
                  for a_idx in range(len(labels))]
        if temp == 0:
            best_action_idx = np.argmax(counts)
            best_action = labels[best_action_idx]
            p = [0] * len(counts)
            p[best_action_idx] = 1
            res['a'] = best_action
            res['pr'] = p
            return res
        else:
            counts = [c**(1. / temp) for c in counts]
            p = [c / float(sum(counts)) for c in counts]
            if self.params.resign_threshold and v > self.params.resign_threshold:
                res['a'] = np.random.choice(labels, p=p)
            res['pr'] = p
            return res

    def _search(self, env):
        
        s = env.to_string()
        c_state = env.canonical_board_state
        
        if s not in self.e_s:
            self.e_s[s] = env.is_game_over
        if self.e_s[s] != 0:
            return -self.e_s[s]

        if s not in self.p_s:
            self.p_s[s], v = self.network.predict(c_state)
            legal = np.asarray(self.env.legal_moves)
            legal_mask = np.isin(labels, legal, assume_unique=True)
            self.p_s[s] = self.p_s[s] * legal_mask
            sum_p_s_s = np.sum(self.p_s[s])
            if sum_p_s_s > 0:
                self.p_s[s] /= sum_p_s_s
            else:
                self.p_s[s] = self.p_s[s] + valids
                self.p_s[s] /= np.sum(self.p_s[s])

            self.v_s[s] = legal_mask
            self.n_s[s] = 0
            return -v

        valids = self.v_s[s]
        cur_best = -float('inf')
        best_action_idx = -1
        for a_idx in range(len(labels)):
            if valids[a_idx]:
                if (s, a_idx) in self.q_sa:
                    u = self.q_sa[(s, a_idx)] + self.params.cpuct * \
                        self.p_s[s][a_idx] * np.sqrt(self.n_s[s]) / \
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
            self.q_sa[(s, best_action_index)] = (
                self.n_sa[(s, best_action_index)] *
                self.q_sq[(s, best_action_index)] + v
            ) / (self.n_sa[(s, best_action_index)] + 1)
            self.n_sa[(s, best_action_index)] += 1
        else:
            self.q_sa[(s, best_action_index)] = v
            self.n_sa[(s, best_action_index)] = 1

        self.n_s[s] += 1
        return -v
        
                         
            

        
        
