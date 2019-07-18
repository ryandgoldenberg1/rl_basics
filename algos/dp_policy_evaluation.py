import gym
from gym import spaces
from gym.envs.toy_text import discrete
import envs


class DpPolicyEvaluation:
    def __init__(self, env):
        assert isinstance(env, discrete.DiscreteEnv)
        self.env = env
        self.reset()

    def evaluate(self, policy, theta=0.001, gamma=1., max_iter=None, verbose=False):
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
                        policy_prob = policy[s][a]
                        new_value += policy_prob * transition_prob * (reward + gamma * self.value_fn[next_state])
                new_value_fn[s] = new_value
                delta = abs(new_value - old_value)
                if delta > max_delta:
                    max_delta = delta
            self.value_fn = new_value_fn
            iter += 1
            if verbose:
                print('iter: {}, value_fn:'.format(iter))
                print(self.value_fn)
            if max_delta < theta:
                break
            if max_iter is not None and iter >= max_iter:
                break
        return self.value_fn


    def reset(self):
        nS = self.env.observation_space.n
        self.value_fn = [0.] * nS


if __name__ == '__main__':
    env = gym.make('GridWorld-v0')
    policy_evaluation = DpPolicyEvaluation(env)
    policy = [ [1/env.nA] * env.nA ] * env.nS
    policy_evaluation.evaluate(policy=policy, max_iter=10, verbose=True)

    value_fn = policy_evaluation.evaluate(policy=policy)
    print('Final Value Fn:')
    print(value_fn)
