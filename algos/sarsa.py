import random
import gym
import matplotlib.pyplot as plt
from tqdm import trange
import envs
from utils.aggregators import ExponentialAverage


class Sarsa:
    def __init__(self, env):
        self.env = env

    def fit(self, discount_factor, step_size, epsilon, num_episodes, verbose=False):
        action_value_fn = [ [0 for _ in range(self.env.action_space.n)] for _ in range(self.env.observation_space.n) ]
        avg_reward = ExponentialAverage()
        timesteps_per_episode = []
        for episode_idx in trange(num_episodes):
            state = self.env.reset()
            action = self._sample_action(action_value_fn, state, epsilon)
            done = False
            total_reward = 0
            step = 0
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self._sample_action(action_value_fn, next_state, epsilon)
                action_value_fn[state][action] += step_size * (reward + discount_factor * action_value_fn[next_state][next_action] - action_value_fn[state][action])
                state = next_state
                action = next_action
                total_reward += (discount_factor ** step) * reward
                step += 1
            avg_reward.add(total_reward)
            timesteps_per_episode.append(step)
            if verbose and episode_idx % 1000 == 0:
                print('Episode: {}, Avg Reward: {}'.format(episode_idx, avg_reward.value()))
        return action_value_fn, timesteps_per_episode


    def _sample_action(self, action_value_fn, state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        action_values = action_value_fn[state]
        max_action_value = max(action_values)
        max_actions = [ i for i in range(len(action_values)) if action_values[i] == max_action_value ]
        assert len(max_actions) > 0, 'No max actions found'
        return random.choice(max_actions)



if __name__ == '__main__':
    discount_factor = 1.
    step_size = 0.1
    epsilon = 0.1
    num_episodes = 170
    verbose = True

    env = gym.make('WindyGridWorld-v0')
    sarsa = Sarsa(env)
    _, timesteps = sarsa.fit(discount_factor=discount_factor, step_size=step_size, epsilon=epsilon, num_episodes=num_episodes, verbose=verbose)

    x = []
    y = []
    num_timesteps = 0
    num_episodes = 0
    for t in timesteps:
        num_timesteps += t
        num_episodes += 1
        x.append(num_timesteps)
        y.append(num_episodes)
    print('avg:', sum(timesteps) / len(timesteps))
    print('avg last 10:', sum(timesteps[-10:]) / 10)
    plt.plot(x, y)
    plt.show()
