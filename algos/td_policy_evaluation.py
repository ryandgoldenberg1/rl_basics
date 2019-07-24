import gym
import math
from tqdm import trange
import envs
import utils.policies as policy_utils


class TdPolicyEvaluation:
    def __init__(self, env):
        self.env = env

    def evaluate(self, policy, num_episodes=10000, step_size=0.1, discount_factor=1.):
        value_fn = [ 0 for _ in range(self.env.observation_space.n) ]
        for episode_idx in trange(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = policy_utils.sample_action(policy, state)
                next_state, reward, done, _ = self.env.step(action)
                value_fn[state] += step_size * (reward + discount_factor * value_fn[next_state] - value_fn[state])
                state = next_state

            # exp_value_fn = [0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14]
            # dist = math.sqrt(sum([(value_fn[i] - exp_value_fn[i])**2 for i in range(len(exp_value_fn))]))
            # if (episode_idx + 1) % 1000 == 0:
            #     print('Episode: {}, Dist: {}'.format(episode_idx, dist))

        return  value_fn


if __name__ == '__main__':
    num_episodes = 100000
    step_size=0.001
    discount_factor=1.

    env = gym.make('GridWorld-v0')
    policy_evaluation = TdPolicyEvaluation(env)
    policy = [ [1/env.nA] * env.nA ] * env.nS
    value_fn = policy_evaluation.evaluate(policy=policy, num_episodes=num_episodes, step_size=step_size, discount_factor=discount_factor)
    print(value_fn)
