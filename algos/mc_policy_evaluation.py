import gym
from gym import spaces
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import utils.policies as policy_utils
from utils.aggregators import IncrementalAverage
import envs.black_jack_wrapper


class McPolicyEvaluation:
    def __init__(self, env):
        assert isinstance(env.observation_space, spaces.Discrete)
        self.env = env

    def evaluate(self, policy, discount_factor=1., num_episodes=100000, first_visit=True):
        value_fn = [IncrementalAverage() for _ in range(self.env.observation_space.n)]
        for _ in trange(num_episodes):
            episode = policy_utils.generate_episode(self.env, policy)
            seen_states = set()
            for element in episode:
                state = element['state']
                element['first_visit'] = (state not in seen_states)
                seen_states.add(state)
            total_reward = 0
            for element in reversed(episode):
                total_reward = discount_factor * total_reward + element['reward']
                if first_visit and not element['first_visit']:
                    continue
                value_fn[element['state']].add(total_reward)
        value_fn = [x.value() for x in value_fn]
        return value_fn


if __name__ == '__main__':
    env = gym.make('DiscreteBlackJack-v0')
    policy = [1] * env.observation_space.n
    for s in range(env.observation_space.n):
        player, dealer, usable_ace = env.reverse_observation(s)
        if player >= 20:
            policy[s] = 0
    evaluator = McPolicyEvaluation(env)
    value_fn = evaluator.evaluate(policy, num_episodes=500000)

    x = np.arange(1, 11)
    y = np.arange(12, 22)
    x, y = np.meshgrid(x, y)
    z1 = np.zeros(x.shape)
    z2 = np.zeros(x.shape)
    for i in range(z1.shape[0]):
        for j in range(z1.shape[1]):
            dealer = x[i][j]
            player = y[i][j]
            v1 = value_fn[ env.observation((player, dealer, True)) ]
            v2 = value_fn[ env.observation((player, dealer, False)) ]
            z1[i][j] = v1
            z2[i][j] = v2

    figure = plt.figure(figsize=plt.figaspect(0.5))
    ax = figure.add_subplot(1, 2, 1, projection='3d')
    ax.plot_wireframe(x, y, z1)
    ax = figure.add_subplot(1, 2, 2, projection='3d')
    ax.plot_wireframe(x, y, z2)
    plt.show()
