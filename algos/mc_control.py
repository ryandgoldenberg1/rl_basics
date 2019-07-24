import gym
import json
import random
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import utils.policies as policy_utils
from utils.aggregators import IncrementalAverage, ExponentialAverage
import envs.black_jack_wrapper


class McControl:
    def __init__(self, env):
        self.env = env

    def fit(self, num_episodes=10000, discount_factor=1., epsilon=0.1, first_visit=True, verbose=False):
        policy = self._random_policy()
        action_value_fn = self._init_action_value_fn()
        avg_reward = ExponentialAverage()
        for episode_idx in trange(num_episodes):
            episode = policy_utils.generate_episode(self.env, policy=policy, epsilon_greedy=True, epsilon=epsilon)
            seen_state_actions = set()
            for event in episode:
                state = event['state']
                action = event['action']
                event['first_visit'] = ((state, action) not in seen_state_actions)
                seen_state_actions.add((state, action))
            total_reward = 0
            for event in reversed(episode):
                total_reward = discount_factor * total_reward + event['reward']
                if first_visit and not event['first_visit']:
                    continue
                action_value_fn[event['state']][event['action']].add(total_reward)
            policy = self._greedy_policy(action_value_fn)
            avg_reward.add(total_reward)
            if verbose:
                print('Episode: {}, Avg Reward: {}'.format(episode_idx, avg_reward.value()))
        action_value_fn = [ [y.value() for y in x] for x in action_value_fn ]
        return action_value_fn, policy

    def _random_policy(self):
        return [ self.env.action_space.sample() for _ in range(self.env.observation_space.n) ]

    def _init_action_value_fn(self):
        return [ [IncrementalAverage() for _ in range(self.env.action_space.n)] for _ in range(self.env.observation_space.n) ]

    def _greedy_policy(self, action_value_fn):
        policy = [-1 for _ in range(len(action_value_fn))]
        for i, action_values in enumerate(action_value_fn):
            action_values = [x.value() for x in action_values]
            max_value = max(action_values)
            max_idxs = [i for i in range(len(action_values)) if action_values[i] == max_value]
            assert len(max_idxs) > 0, 'Could not find max action'
            max_action = random.choice(max_idxs)
            policy[i] = max_action
        assert all([ x >= 0 for x in policy ])
        return policy


if __name__ == '__main__':
    env = gym.make('DiscreteBlackJack-v0')
    mc_control = McControl(env)
    action_value_fn, policy = mc_control.fit(num_episodes=5000, discount_factor=1, epsilon=0.1, first_visit=True, verbose=False)
    state_value_fn = [ max(action_values) for action_values in action_value_fn ]

    with open('blackjack_policy.json', 'w') as f:
        json.dump(policy, f)

    x = np.arange(1, 11)
    y = np.arange(12, 22)
    x, y = np.meshgrid(x, y)
    z1 = np.zeros(x.shape)
    z2 = np.zeros(x.shape)
    z3 = np.zeros(x.shape)
    z4 = np.zeros(x.shape)
    for i in range(z1.shape[0]):
        for j in range(z1.shape[1]):
            dealer = x[i][j]
            player = y[i][j]
            z1[i][j] = state_value_fn[ env.observation((player, dealer, True)) ]
            z2[i][j] = state_value_fn[ env.observation((player, dealer, False)) ]
            z3[i][j] = policy[env.observation((player, dealer, True))]
            z4[i][j] = policy[env.observation((player, dealer, False))]

    figure = plt.figure(figsize=plt.figaspect(0.5))
    ax = figure.add_subplot(2, 2, 1, projection='3d')
    ax.plot_wireframe(x, y, z3)
    ax = figure.add_subplot(2, 2, 2, projection='3d')
    ax.plot_wireframe(x, y, z1)
    ax = figure.add_subplot(2, 2, 3, projection='3d')
    ax.plot_wireframe(x, y, z4)
    ax = figure.add_subplot(2, 2, 4, projection='3d')
    ax.plot_wireframe(x, y, z2)
    plt.show()
