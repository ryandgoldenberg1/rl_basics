from collections import defaultdict
import copy
import random
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


if __name__ == '__main__':
    register(id='Easy21-v0', entry_point='exercises.easy21:Easy21Env')
    env = gym.make('Easy21-v0')

    # Part 1: Implementation of Easy21
    # play_text('Easy21-v0')


    # Part 2: Monte Carlo Control
    mc_control = MonteCarloControl(env, n0=100, discount_factor=1)
    mc_control.train(num_episodes=1000000)
    mc_control.plot_value_fn()
