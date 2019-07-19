import gym
from gym import spaces
from gym.envs.toy_text import discrete
import numpy as np
import envs



def _get_policy_prob(policy, state, action):
    if isinstance(policy, dict) and state not in policy:
        return 0
    dist = policy[state]
    if isinstance(dist, int):
        return 1 if dist == action else 0
    if isinstance(dist, dict) and action not in dist:
        return 0
    return dist[action]



class DpIterativePolicyEvaluation:
    def __init__(self, env):
        assert isinstance(env, discrete.DiscreteEnv)
        self.env = env
        self.reset()

    def evaluate(self, policy, theta=0.001, gamma=1., max_iter=None, verbose=False, in_place=False):
        self.reset()
        iter = 0
        while True:
            max_delta = 0
            new_value_fn = [0.] * self.env.nS
            for s in range(self.env.nS):
                if s not in self.env.P:
                    continue
                old_value = self.value_fn[s]
                new_value = 0
                for a in self.env.P[s].keys():
                    for transition_prob, next_state, reward, _ in self.env.P[s][a]:
                        policy_prob = _get_policy_prob(policy=policy, state=s, action=a)
                        new_value += policy_prob * transition_prob * (reward + gamma * self.value_fn[next_state])
                new_value_fn[s] = new_value
                if in_place:
                    self.value_fn[s] = new_value
                delta = abs(new_value - old_value)
                if delta > max_delta:
                    max_delta = delta
            if not in_place:
                self.value_fn = new_value_fn
            iter += 1
            if verbose:
                print('iter: {}, value_fn:'.format(iter))
                print(self.value_fn)
            if max_delta <= theta:
                break
            if max_iter is not None and iter >= max_iter:
                break
        return self.value_fn

    def reset(self):
        nS = self.env.observation_space.n
        self.value_fn = [0.] * nS


class DpLinearPolicyEvaluation:
    def __init__(self, env):
        assert isinstance(env, discrete.DiscreteEnv)
        self.env = env

    def evaluate(self, policy, gamma=1.):
        # P[s][a] => [ (prob, next-state, reward, done) ]
        # V(s) = sum_a policy(s, a) sum_s' P(s, s', a) [ R(s, s', a) + gamma * V(s') ]
        # Derive system of equations Ax = b, where x = V(s) from above.
        # A: n x n
        nS = self.env.nS
        nA = self.env.nA
        A = np.zeros((nS, nS))
        b = np.zeros(nS)

        for s in range(nS):
            A[s][s] -= 1
            for a in range(nA):
                if a not in self.env.P[s]:
                    continue
                policy_prob = _get_policy_prob(policy=policy, state=s, action=a)
                for transition_prob, next_state, reward, _ in self.env.P[s][a]:
                    b[s] -= policy_prob * transition_prob * reward
                    A[s][next_state] += policy_prob * transition_prob * gamma
        V = np.linalg.solve(A, b)
        return V.tolist()


if __name__ == '__main__':
    env = gym.make('GridWorld-v0')
    policy_evaluation = DpIterativePolicyEvaluation(env)
    policy = [ [1/env.nA] * env.nA ] * env.nS
    policy_evaluation.evaluate(policy=policy, max_iter=10, verbose=True)

    value_fn = policy_evaluation.evaluate(policy=policy, theta=0., verbose=False)
    print('Final Value Fn:')
    print(value_fn)

    linear_policy_solver = DpLinearPolicyEvaluation(env)
    linear_value_fn = linear_policy_solver.evaluate(policy)
    print('Solved Value Fn:')
    print(linear_value_fn)
