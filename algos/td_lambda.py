import gym
from tqdm import trange
import math
import utils.policies as policy_utils
import envs


class EligibilityTrace:
    def __init__(self, num_states, lambda_, discount_factor):
        self.num_states = num_states
        self.lambda_ = lambda_
        self.discount_factor = discount_factor
        self.values = [0 for _ in range(num_states)]

    def update(self, state):
        for s in range(self.num_states):
            self.values[s] *= self.lambda_ * self.discount_factor
        self.values[state] += 1

    def get(self, state):
        return self.values[state]


class TdLambda:
    def __init__(self, env):
        self.env = env

    def evaluate(self, policy, num_episodes=10000, step_size=0.1, discount_factor=1., lambda_=0.9):
        num_states = self.env.observation_space.n
        value_fn = [ 0 for _ in range(self.env.observation_space.n) ]
        for episode_idx in trange(num_episodes):
            state = self.env.reset()
            done = False
            eligibility_trace = EligibilityTrace(num_states=num_states, lambda_=lambda_, discount_factor=discount_factor)
            while not done:
                action = policy_utils.sample_action(policy, state)
                next_state, reward, done, _ = self.env.step(action)
                eligibility_trace.update(state)
                delta = reward + discount_factor * value_fn[next_state] - value_fn[state]
                for s in range(num_states):
                    e = eligibility_trace.get(s)
                    value_fn[s] += step_size * delta * e
                state = next_state

            # exp_value_fn = [0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14]
            # dist = math.sqrt(sum([(value_fn[i] - exp_value_fn[i])**2 for i in range(len(exp_value_fn))]))
            # if (episode_idx + 1) % 1000 == 0:
            #     print('Episode: {}, Dist: {}'.format(episode_idx, dist))
        return value_fn


if __name__ == '__main__':
    num_episodes = 100000
    step_size = 0.0001
    discount_factor = 1.
    lambda_ = 0.9

    env = gym.make('GridWorld-v0')
    td_lambda = TdLambda(env)
    policy = [ [1/env.nA] * env.nA ] * env.nS
    value_fn = td_lambda.evaluate(policy, num_episodes=num_episodes, step_size=step_size,
                                  discount_factor=discount_factor, lambda_=lambda_)
    print(value_fn)
