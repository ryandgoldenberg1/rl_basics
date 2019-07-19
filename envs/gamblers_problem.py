import argparse
import random
from gym.envs.toy_text import discrete
from utils.play_text import play_text


class GamblersProblemEnv(discrete.DiscreteEnv):
    def __init__(self, p=0.5):
        assert p > 0 and p < 1
        print('p:', p)
        nS = 101
        nA = 100
        P = {}
        for s in range(nS):
            P[s] = {}
            for a in range(nA):
                if a > s or s >= 100 or a == 0:
                    P[s][a] = []
                else:
                    win_prob = p
                    win_result = min(s + a, 100)
                    win_reward = 1 if s + a >= 100 else 0
                    win_done = (win_result == 100)

                    loss_prob = 1 - p
                    loss_result = s - a
                    loss_reward = 0
                    loss_done = (s == a)

                    P[s][a] = [
                        (win_prob, win_result, win_reward, win_done),
                        (loss_prob, loss_result, loss_reward, loss_done),
                    ]
        isd = [0] * nS
        isd[50] = 1.
        super().__init__(nS=nS, nA=nA, P=P, isd=isd)

    def render(self, mode='human'):
        print('Bankroll: ${}'.format(self.s))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=0.5)
    args = parser.parse_args()
    play_text(env_id='GamblersProblem-v0', p=args.p)
