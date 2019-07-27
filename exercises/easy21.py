import random
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


if __name__ == '__main__':
    register(id='Easy21-v0', entry_point='exercises.easy21:Easy21Env')

    # Part 1: Implementation of Easy21
    # play_text('Easy21-v0')
