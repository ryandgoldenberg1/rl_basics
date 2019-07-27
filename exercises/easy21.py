from collections import defaultdict
import copy
import json
import random
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import gym
from gym import spaces
from gym.envs.registration import register
from utils.play_text import play_text


class Easy21Env(gym.Env):
    # http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf
    #
    # • The game is played with an infinite deck of cards (i.e. cards are sampled
    # with replacement)
    # • Each draw from the deck results in a value between 1 and 10 (uniformly
    # distributed) with a colour of red (probability 1/3) or black (probability
    # 2/3).
    # • There are no aces or picture (face) cards in this game
    # • At the start of the game both the player and the dealer draw one black
    # card (fully observed)
    # • Each turn the player may either stick or hit
    # • If the player hits then she draws another card from the deck
    # • If the player sticks she receives no further cards
    # • The values of the player’s cards are added (black cards) or subtracted (red
    # cards)
    # • If the player’s sum exceeds 21, or becomes less than 1, then she “goes
    # bust” and loses the game (reward -1)
    # • If the player sticks then the dealer starts taking turns. The dealer always
    # sticks on any sum of 17 or greater, and hits otherwise. If the dealer goes
    # bust, then the player wins; otherwise, the outcome – win (reward +1),
    # lose (reward -1), or draw (reward 0) – is the player with the largest sum.
    #
    # The state is represented as a tuple (player_sum, dealer_sum)
    # Actions are 0 for stick, 1 for hit

    def __init__(self):
        self.observation_space = spaces.Tuple((spaces.Discrete(41), spaces.Discrete(41)))
        self.num_states = 41 * 41
        self.action_space = spaces.Discrete(2)
        self.reset()

    def reset(self):
        self.player_sum = abs(self._draw_card())
        self.dealer_sum = abs(self._draw_card())
        self.last_action = None
        self.done = False
        return self._get_obs()

    def step(self, action):
        assert action in self.action_space
        if action == 0:
            self._play_dealer()
            self.done = True
        elif action == 1:
            self.player_sum += self._draw_card()
            if self._is_bust(self.player_sum):
                self.done = True
        self.last_action = action
        return self._get_obs(), self._get_reward(), self.done, None

    def render(self, mode='human'):
        last_action = 'None' if self.last_action is None else ['Stick', 'Hit'][self.last_action]
        print('(' + last_action + ')')
        print('Player:', self.player_sum)
        print('Dealer:', self.dealer_sum)

    def _draw_card(self):
        value = random.choice(range(1, 11))
        if random.choice(range(3)) == 0:
            value *= -1
        return value

    def _is_bust(self, total):
        return total > 21 or total < 1

    def _play_dealer(self):
        while self.dealer_sum < 17 and not self._is_bust(self.dealer_sum):
            self.dealer_sum += self._draw_card()

    def _get_obs(self):
        return (self.player_sum + 9, self.dealer_sum + 9)

    def _get_reward(self):
        if not self.done:
            return 0
        if self._is_bust(self.player_sum):
            return -1
        if self._is_bust(self.dealer_sum):
            return 1
        if self.player_sum < self.dealer_sum:
            return -1
        if self.player_sum == self.dealer_sum:
            return 0
        if self.player_sum > self.dealer_sum:
            return 1

    def state_to_id(self, state):
        assert state in self.observation_space
        return state[0] * self.observation_space[1].n + state[1]

    def id_to_state(self, state_id):
        return (int(state_id / self.observation_space[1].n), state_id % self.observation_space[1].n)

    def sums_to_state_id(self, player_sum, dealer_sum):
        state = (player_sum + 9, dealer_sum + 9)
        return self.state_to_id(state)

    def state_id_to_sums(self, state_id):
        state = self.id_to_state(state_id)
        player_sum, dealer_sum = state
        return (player_sum - 9, dealer_sum - 9)



class MonteCarloControl:
    def __init__(self, env, n0, discount_factor):
        num_states = env.num_states
        num_actions = env.action_space.n
        self.env = env
        self.n0 = n0
        self.discount_factor = discount_factor
        self.action_value_fn = [[0 for _ in range(num_actions)] for _ in range(num_states)]
        self.action_visits = [[0 for _ in range(num_actions)] for _ in range(num_states)]
        self.state_visits = [0 for _ in range(num_states)]
        self.num_states = num_states
        self.num_actions = num_actions

    def train(self, num_episodes):
        prev_action_value_fn = copy.deepcopy(self.action_value_fn)
        for episode_idx in range(num_episodes):
            episode = self._generate_episode()
            total_reward = 0
            for event in reversed(episode):
                total_reward = total_reward * self.discount_factor + event['reward']
                state = event['state']
                action = event['action']
                state_id = self.env.state_to_id(state)
                self.action_visits[state_id][action] += 1
                step_size = 1 / self.action_visits[state_id][action]
                delta = total_reward - self.action_value_fn[state_id][action]
                self.action_value_fn[state_id][action] += step_size * delta
            episode_num = episode_idx + 1
            if (episode_num % 10000) == 0:
                avg_reward = self.evaluate(num_episodes=1000)
                value_fn_diff = 0
                for s in range(self.num_states):
                    for a in range(self.num_actions):
                        value_fn_diff += abs(self.action_value_fn[s][a] - prev_action_value_fn[s][a])
                prev_action_value_fn = copy.deepcopy(self.action_value_fn)
                print('Episode: {}, AvgReward: {}, ValueFnDiff: {} '.format(episode_num, avg_reward, value_fn_diff))

    def evaluate(self, num_episodes):
        total = 0
        for _ in range(num_episodes):
            state = self.env.reset()
            state_id = self.env.state_to_id(state)
            done = False
            step = 0
            while not done:
                action = self._greedy_action(state_id)
                state, reward, done, _ = self.env.step(action)
                state_id = self.env.state_to_id(state)
                step += 1
                total += (self.discount_factor ** step) * reward
        return total / num_episodes

    def plot_value_fn(self):
        x = np.arange(1, 11)
        y = np.arange(1, 11)
        x, y = np.meshgrid(x, y)
        z = np.zeros(x.shape)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                player_sum = x[i][j]
                dealer_sum = y[i][j]
                state_id = self.env.sums_to_state_id(player_sum, dealer_sum)
                value = max(self.action_value_fn[state_id])
                z[i][j] = value
        figure = plt.figure(figsize=plt.figaspect(0.5))
        ax = figure.add_subplot(1, 1, 1, projection='3d')
        ax.plot_wireframe(x, y, z)
        plt.show()

    def _generate_episode(self):
        episode = []
        state = self.env.reset()
        state_id = self.env.state_to_id(state)
        done = False
        while not done:
            self.state_visits[state_id] += 1
            action = self._act(state_id)
            next_state, reward, done, _ = self.env.step(action)
            episode.append({'state': state, 'action': action, 'reward': reward, 'next_state': next_state})
            state = next_state
            state_id = self.env.state_to_id(state)
        return episode

    def _act(self, state_id):
        epsilon = self.n0 / (self.n0 + self.state_visits[state_id])
        if random.random() < epsilon:
            return self.env.action_space.sample()
        return self._greedy_action(state_id)

    def _greedy_action(self, state_id):
        action_values = self.action_value_fn[state_id]
        max_action_value = max(action_values)
        max_actions = [i for i, value in enumerate(action_values) if value == max_action_value]
        return random.choice(max_actions)


class SarsaLambda:
    def __init__(self, env, n0, discount_factor, lambda_):
        num_states = env.num_states
        num_actions = env.action_space.n

        self.num_states = num_states
        self.num_actions = num_actions
        self.env = env
        self.n0 = n0
        self.discount_factor = discount_factor
        self.lambda_ = lambda_
        self.action_value_fn = [[0 for _ in range(num_actions)] for _ in range(num_states)]
        self.state_visits = [0 for _ in range(num_states)]
        self.action_visits = [[0 for _ in range(num_actions)] for _ in range(num_states)]

    def train(self, num_episodes):
        for episode_idx in range(num_episodes):
            state = self.env.reset()
            state_id = self.env.state_to_id(state)
            self.state_visits[state_id] += 1
            action = self._act(state_id)
            done = False
            eligibility_trace = [[0 for _ in range(self.num_actions)] for _ in range(self.num_states)]
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_state_id = self.env.state_to_id(next_state)
                for s in range(self.num_states):
                    for a in range(self.num_actions):
                        eligibility_trace[s][a] *= self.discount_factor * self.lambda_
                eligibility_trace[state_id][action] += 1
                self.action_visits[state_id][action] += 1
                self.state_visits[next_state_id] += 1
                next_action = self._act(next_state_id)
                step_size = 1 / self.action_visits[state_id][action]
                delta = reward + self.action_value_fn[next_state_id][next_action] - self.action_value_fn[state_id][action]
                self.action_value_fn[state_id][action] += step_size * delta * eligibility_trace[state_id][action]
                state = next_state
                state_id = next_state_id
                action = next_action

    def _act(self, state_id):
        epsilon = self.n0 / (self.n0 + self.state_visits[state_id])
        if random.random() < epsilon:
            return self.env.action_space.sample()
        return self._greedy_action(state_id)

    def _greedy_action(self, state_id):
        action_values = self.action_value_fn[state_id]
        max_action_value = max(action_values)
        max_actions = [i for i, value in enumerate(action_values) if value == max_action_value]
        return random.choice(max_actions)



class ApproxSarsaLambda:
    def __init__(self, env, n0, discount_factor, lambda_, epsilon, step_size):
        num_states = env.num_states
        num_actions = env.action_space.n
        self.num_states = num_states
        self.num_actions = num_actions
        self.env = env
        self.n0 = n0
        self.discount_factor = discount_factor
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.step_size = step_size

        cuboids = []
        for dealer_range in [range(1, 5), range(4, 8), range(7, 11)]:
            for player_range in [range(1, 7), range(4, 10), range(7, 13), range(10, 16), range(13, 19), range(16, 22)]:
                for action in range(num_actions):
                    cuboids.append((dealer_range, player_range, action))
        self.cuboids = cuboids
        self.num_features = len(cuboids)
        self.weights = [0 for _ in range(self.num_features)]

    def feature_fn(self, state_id, action):
        player_sum, dealer_sum = self.env.state_id_to_sums(state_id)
        features = [0 for _ in range(self.num_features)]
        for i, cuboid in enumerate(self.cuboids):
            if dealer_sum in cuboid[0] and player_sum in cuboid[1] and action == cuboid[2]:
                features[i] = 1
        return features

    def train(self, num_episodes):
        for episode_idx in trange(num_episodes):
            state = self.env.reset()
            state_id = self.env.state_to_id(state)
            done = False
            action = self._act(state_id)
            eligibility_trace = [0 for _ in range(self.num_features)]
            while not done:
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                next_state_id = self.env.state_to_id(next_state)
                next_action = self._act(next_state_id)

                # Update eligibility trace
                eligibility_trace = [x * self.discount_factor * self.lambda_ for x in eligibility_trace]
                features = self.feature_fn(state_id, action)
                for i in range(self.num_features):
                    eligibility_trace[i] += features[i]

                # Perform sarsa update
                target = reward + self.discount_factor * self._action_value(next_state_id, next_action)
                delta = target - self._action_value(state_id, action)
                for i in range(self.num_features):
                    self.weights[i] += self.step_size * delta * eligibility_trace[i]

                # Go to next step
                state = next_state
                state_id = next_state_id
                action = next_action

    def action_value_fn(self):
        action_value_fn = [[0 for _ in range(self.num_actions)] for _ in range(self.num_states)]
        for state in range(self.num_states):
            for action in range(self.num_actions):
                value = self._action_value(state, action)
                action_value_fn[state][action] = value
        return action_value_fn

    def _act(self, state_id):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self._greedy_action(state_id)

    def _action_value(self, state_id, action):
        features = self.feature_fn(state_id, action)
        return sum(f * w for f, w in zip(features, self.weights))

    def _greedy_action(self, state_id):
        action_values = [self._action_value(state_id, action) for action in range(self.num_actions)]
        max_action_value = max(action_values)
        max_actions = [i for i, value in enumerate(action_values) if value == max_action_value]
        return random.choice(max_actions)


if __name__ == '__main__':
    register(id='Easy21-v0', entry_point='exercises.easy21:Easy21Env')
    env = gym.make('Easy21-v0')

    # Part 1: Implementation of Easy21
    # play_text('Easy21-v0')


    # Part 2: Monte Carlo Control
    # mc_control = MonteCarloControl(env, n0=100, discount_factor=1)
    # mc_control.train(num_episodes=1000000)
    # mc_control.plot_value_fn()
    # with open('mc_value_fn.json', 'w') as f:
    #     json.dump(mc_control.action_value_fn, f)

    # with open('mc_value_fn.json') as f:
    #     mc_value_fn = json.load(f)
    # def _mse(action_value_fn):
    #     value_diff = 0
    #     num_states = len(mc_value_fn)
    #     num_actions = env.action_space.n
    #     for s in range(num_states):
    #         for a in range(num_actions):
    #             value_diff += (mc_value_fn[s][a] - action_value_fn[s][a]) ** 2
    #     value_diff /= (num_states * num_actions)
    #     return value_diff
    # lambdas = np.arange(0, 1.1, 0.1).tolist()

    # Part 3: Sarsa
    # value_diffs = []
    # for lambda_ in tqdm(lambdas):
    #     sarsa = SarsaLambda(env, n0=100, discount_factor=1, lambda_=lambda_)
    #     sarsa.train(num_episodes=10000)
    #     value_diffs.append(_mse(sarsa.action_value_fn))
    # plt.plot(lambdas, value_diffs)
    # plt.show()
    #
    # for lambda_ in (0, 1):
    #     sarsa = SarsaLambda(env, n0=100, discount_factor=1, lambda_=lambda_)
    #     y = [_mse(sarsa.action_value_fn)]
    #     x = [0]
    #     for _ in trange(20):
    #         sarsa.train(num_episodes=100)
    #         y.append(_mse(sarsa.action_value_fn))
    #         x.append(x[-1] + 100)
    #     plt.plot(x, y)
    #     plt.show()


    # Part 4: Linear Function Approximation in Easy21
    # sarsa_approx = ApproxSarsaLambda(env, n0=100, discount_factor=1, lambda_=1., epsilon=0.05, step_size=0.01)
    # x = [0]
    # sarsa_approx.action_value_fn()
    # y = [_mse(sarsa_approx.action_value_fn()) ]
    # for _ in range(20):
    #     sarsa_approx.train(num_episodes=100)
    #     x.append(x[-1] + 100)
    #     y.append(_mse(sarsa_approx.action_value_fn()))
    # plt.plot(x, y)
    # plt.show()
    #
    # y = []
    # for lambda_ in tqdm(lambdas):
    #     sarsa_approx = ApproxSarsaLambda(env, n0=100, discount_factor=1, lambda_=1., epsilon=0.05, step_size=0.01)
    #     sarsa_approx.train(num_episodes=10000)
    #     action_value_fn = sarsa_approx.action_value_fn()
    #     y.append(_mse(action_value_fn))
    # plt.plot(lambdas, y)
    # plt.show()
