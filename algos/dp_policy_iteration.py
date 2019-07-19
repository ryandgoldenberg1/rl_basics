import gym
import random
import pprint as pp
from gym.envs.toy_text import discrete
from .dp_policy_evaluation import DpIterativePolicyEvaluation


class DpPolicyIteration:
    def __init__(self, env, discount_factor=1.):
        assert isinstance(env, discrete.DiscreteEnv)
        assert discount_factor >= 0 and discount_factor <= 1
        self.env = env
        self.discount_factor = discount_factor
        self.policy_evaluation = DpIterativePolicyEvaluation(env)

    def fit(self, verbose=False):
        # Initialize random policy and value function
        policy = [ [1/self.env.nA] * self.env.nA for _ in range(self.env.nS) ]
        value_fn = [0] * self.env.nS
        iter = 0
        while True:
            value_fn = self.policy_evaluation.evaluate(policy, gamma=self.discount_factor, in_place=True)
            new_policy = self._greedy_policy(value_fn)
            if new_policy == policy:
                break
            policy = new_policy
            iter += 1
            if verbose:
                print('Iter: ', iter)
                print('Value Function:')
                print(value_fn)
                print('Policy: ')
                pp.pprint(policy)

        return value_fn, policy

    def _greedy_policy(self, value_fn):
        policy = [ [0] * self.env.nA for _ in range(self.env.nS) ]
        for s in range(self.env.nS):
            best_action = None
            best_action_value = -float('inf')
            for a in range(self.env.nA):
                if a not in self.env.P[s]:
                    continue
                a_value = 0.
                for prob, next_state, reward, _ in self.env.P[s][a]:
                    a_value += prob * (reward + self.discount_factor * value_fn[next_state])
                if a_value > best_action_value:
                    best_action = a
                    best_action_value = a_value
            assert best_action is not None
            policy[s][best_action] = 1.
        return policy


if __name__ == '__main__':
    env = gym.make('GridWorld-v0')
    policy_iteration = DpPolicyIteration(env, discount_factor=0.9)
    policy_iteration.fit(verbose=True)
