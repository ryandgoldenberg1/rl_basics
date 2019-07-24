import random
import gym
from tqdm import trange
import matplotlib.pyplot as plt


class QLearning:
    def __init__(self, env):
        self.env = env

    def fit(self, num_episodes, step_size, discount_factor, epsilon):
        action_value_fn = [ [0 for _ in range(self.env.action_space.n)] for _ in range(self.env.observation_space.n) ]
        episode_results = []
        for episode_idx in trange(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            step = 0
            while not done:
                action = self._sample_action(action_value_fn, state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                action_value_fn[state][action] += step_size * (reward + discount_factor * max(action_value_fn[next_state]) - action_value_fn[state][action])
                state = next_state
                total_reward += (discount_factor ** step) * reward
                step += 1
            episode_results.append(total_reward)
        return action_value_fn, episode_results


    def _sample_action(self, action_value_fn, state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        action_values = action_value_fn[state]
        max_action_value = max(action_values)
        max_actions = [ i for i in range(len(action_values)) if action_values[i] == max_action_value ]
        return random.choice(max_actions)


def _exp_smooth(values, factor):
    initial_value = sum(values[:10]) / 10
    result = []
    curr = initial_value
    for value in values:
        curr = curr + factor * (value - curr)
        result.append(curr)
    return result


if __name__ == '__main__':
    num_episodes = 500
    step_size = 0.1
    epsilon = 0.1
    discount_factor = 1.
    smoothing_factor = 0.02

    env = gym.make('CliffWalking-v0')
    q_learning = QLearning(env)
    _, rewards = q_learning.fit(num_episodes=num_episodes, step_size=step_size, discount_factor=discount_factor, epsilon=epsilon)

    rewards_smooth = _exp_smooth(rewards, smoothing_factor)
    plt.plot(range(len(rewards)), rewards_smooth)
    plt.ylim(-100, -25)
    plt.show()
