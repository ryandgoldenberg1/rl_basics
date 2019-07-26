import gym
import random
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


class EligibilityTrace:
    def __init__(self, num_features, discount_factor, lambda_, replacing):
        self.num_features = num_features
        self.discount_factor = discount_factor
        self.lambda_ = lambda_
        self.replacing = replacing
        self.values = [0] * num_features

    def get(self, i):
        return self.values[i]

    def update(self, features):
        assert len(features) == self.num_features
        self.decay()
        if self.replacing:
            self.replace(features)
        for i, f in enumerate(features):
            if f > 0:
                self.values[i] += 1

    def decay(self):
        decay_factor = self.discount_factor * self.lambda_
        self.values = [ x * decay_factor for x in self.values ]

    def replace(self, features):
        for i, f in enumerate(features):
            if f > 0:
                self.values[i] = 0


class SarsaLambdaLinear:
    def __init__(self, env, feature_fn):
        self.env = env
        self.feature_fn = feature_fn

    def fit(self, num_episodes, step_size, discount_factor, epsilon, lambda_, replacing, log_episodes, eval_episodes):
        num_features = len(self.feature_fn)
        weights = [0] * num_features
        for episode_idx in trange(num_episodes):
            state = self.env.reset()
            done = False
            eligibility_trace = EligibilityTrace(num_features, discount_factor, lambda_, replacing)
            action = self._sample_action(weights, state, epsilon)
            while not done:
                features = self.feature_fn(state, action)
                eligibility_trace.update(features)
                next_state, reward, done, _ = self.env.step(action)
                max_action_value = self._max_action_value(weights, next_state)
                action_value = self._action_value(weights, state, action)
                delta = reward + discount_factor * max_action_value - action_value
                for i in range(num_features):
                    weights[i] += step_size * delta * eligibility_trace.get(i)
                state = next_state
                action = self._sample_action(weights, state, epsilon)
            episode_num = episode_idx + 1
            if episode_num % log_episodes == 0:
                avg_reward = self.evaluate(weights, discount_factor, eval_episodes)
                print('Episode: {}, Avg Reward: {}'.format(episode_num, avg_reward))
        return weights

    def evaluate(self, weights, discount_factor, eval_episodes):
        episode_rewards = []
        for _ in trange(eval_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            step = 0
            while not done:
                action = self.act(weights, state)
                state, reward, done, _ = self.env.step(action)
                episode_reward += (discount_factor ** step) * reward
                step += 1
            episode_rewards.append(episode_reward)
        return sum(episode_rewards) / len(episode_rewards)

    def act(self, weights, state):
        action_values = [ self._action_value(weights, state, action) for action in range(self.env.action_space.n) ]
        max_action_value = max(action_values)
        max_actions = [ i for i, v in enumerate(action_values) if v == max_action_value ]
        return random.choice(max_actions)

    def _sample_action(self, weights, state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        action_values = [ self._action_value(weights, state, action) for action in range(self.env.action_space.n) ]
        max_action_value = max(action_values)
        max_actions = [ i for i, value in enumerate(action_values) if value == max_action_value ]
        return random.choice(max_actions)

    def _action_value(self, weights, state, action):
        features = self.feature_fn(state, action)
        return sum(w * f for w, f in zip(weights, features))

    def _max_action_value(self, weights, state):
        action_values = [ self._action_value(weights, state, action) for action in range(self.env.action_space.n) ]
        return max(action_values)


class Tiling2D:
    def __init__(self, width, height, x_left, y_top, num_rows, num_cols):
        self.width = width
        self.height = height
        self.x_left = x_left
        self.y_top = y_top
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.x_right = x_left + width * num_cols
        self.y_bottom = y_top - height * num_rows

    def __call__(self, x, y):
        if x < self.x_left or x >= self.x_right or y <= self.y_bottom or y > self.y_top:
            return -1
        x_offset = x - self.x_left
        col_offset = int(x_offset / self.width)
        y_offset = self.y_top - y
        row_offset = int(y_offset / self.height)
        return row_offset * self.num_cols + col_offset

    def __len__(self):
        return self.num_rows * self.num_cols


class MountainCarFeatureFn:
    def __init__(self, env):
        x_low, y_low = env.observation_space.low.tolist()
        x_high, y_high = env.observation_space.high.tolist()
        num_actions = env.action_space.n
        tilings = []
        tile_width = 1.8 / 8
        tile_height = 0.14 / 8
        num_tilings = 10
        num_rows = 9
        num_cols = 9
        for _ in range(num_tilings):
            x_rand = random.random()
            x_left = x_low - x_rand * tile_width
            y_rand = random.random()
            y_top = y_high + y_rand * tile_height
            tiling = Tiling2D(width=tile_width, height=tile_height, x_left=x_left, y_top=y_top, num_rows=num_rows, num_cols=num_cols)
            tilings.append(tiling)
        self.tilings = tilings
        self.num_tiling_features = sum(len(x) for x in self.tilings)
        self.num_features = self.num_tiling_features * num_actions

    def __call__(self, state, action):
        result = [0] * self.num_features
        start_idx = action * self.num_tiling_features
        x, y = state.tolist()
        for tiling in self.tilings:
            tiling_id = tiling(x, y)
            if tiling_id >= 0:
                result[start_idx + tiling_id] = 1
            start_idx += len(tiling)
        return result

    def __len__(self):
        return self.num_features


if __name__ == '__main__':
    num_episodes = 1000
    log_episodes = 100
    eval_episodes = 100
    lambda_ = 0.9
    epsilon = 0.
    discount_factor = 1.
    step_size = 0.05
    replacing = True

    env = gym.make('MountainCar-v0')
    feature_fn = MountainCarFeatureFn(env)
    sarsa = SarsaLambdaLinear(env, feature_fn)
    weights = sarsa.fit(num_episodes=num_episodes, step_size=step_size, discount_factor=discount_factor, epsilon=epsilon,
              lambda_=lambda_, replacing=replacing, log_episodes=log_episodes, eval_episodes=eval_episodes)

    x = np.arange(-1.2, 0.7, 0.1)
    y = np.arange(-0.07, 0.08, 0.01)
    x, y = np.meshgrid(x, y)
    z = np.zeros(x.shape)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            x_val = x[i][j]
            y_val = y[i][j]
            state = np.array([x_val, y_val])
            z[i][j] = -sarsa._max_action_value(weights, state)
    figure = plt.figure(figsize=plt.figaspect(0.5))
    ax = figure.add_subplot(1, 1, 1, projection='3d')
    ax.plot_wireframe(x, y, z)
    plt.show()
