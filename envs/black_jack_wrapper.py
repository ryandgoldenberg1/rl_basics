import gym
from gym import spaces
from gym.envs.toy_text import BlackjackEnv
from utils.play_text import play_text


class BlackJackWrapperEnv(gym.ObservationWrapper):
    # Discrete(32), Discrete(11), Discrete(2) -> Discrete(704)
    def __init__(self):
        env = BlackjackEnv()
        super().__init__(env)
        self.observation_space = spaces.Discrete(704)

    def observation(self, observation):
        state = 0
        state += observation[0]
        state += observation[1] * 32
        state += int(observation[2]) * 11 * 32
        return state

    def reverse_observation(self, observation):
        usable_ace = bool(int(observation / (11 * 32)))
        observation = observation % (11 * 32)
        dealer = int(observation / 32)
        observation = observation % 32
        player = observation
        return (player, dealer, usable_ace)

    def render(self):
        player, dealer, usable_ace = self.env._get_obs()
        print('Player: {}, Dealer: {}, Usable Ace: {}'.format(player, dealer, usable_ace))


if __name__ == '__main__':
    play_text(env_id='DiscreteBlackJack-v0')
