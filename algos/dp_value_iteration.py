import gym
from gym.envs.toy_text import discrete
import matplotlib.pyplot as plt
import envs.gamblers_problem



class DpValueIteration:
    def __init__(self, env, discount_factor=1.):
        assert isinstance(env, discrete.DiscreteEnv)
        assert discount_factor >= 0 and discount_factor <= 1
        self.env = env
        self.discount_factor = discount_factor

    def fit(self, theta=0.001, return_history=False):
        value_fn = [0] * self.env.nS
        history = [value_fn]
        iter = 0
        while True:
            new_value_fn = [0] * self.env.nS
            for s in range(self.env.nS):
                best_a_value = -float('inf')
                best_a = None
                for a in range(self.env.nA):
                    a_value = 0
                    for prob, next_state, reward, _ in self.env.P[s][a]:
                        a_value += prob * (reward + self.discount_factor * value_fn[next_state])
                    if a_value > best_a_value:
                        best_a_value = a_value
                        best_a = a
                if best_a_value == -float('inf'):
                    best_a_value = 0.
                new_value_fn[s] = best_a_value
            value_fn_diff = max([  abs(value_fn[i] - new_value_fn[i]) for i in range(self.env.nS) ])
            value_fn = new_value_fn
            iter += 1
            print('iter: {}, diff: {}'.format(iter, value_fn_diff))
            if return_history:
                history.append(value_fn)
            if value_fn_diff <= theta:
                break

        policy = [0] * self.env.nS
        for s in range(self.env.nS):
            for a in range(self.env.nA):
                a_value = 0
                for prob, next_state, reward, _ in self.env.P[s][a]:
                    a_value += prob * (reward + self.discount_factor * value_fn[next_state])
                if a_value >= value_fn[s]:
                    policy[s] = a

        if return_history:
            return value_fn, policy, history
        return value_fn


if __name__ == '__main__':
    env = gym.make('GamblersProblem-v0', p=0.4)
    value_iteration = DpValueIteration(env, discount_factor=1.)
    value_fn, policy, history = value_iteration.fit(theta=0., return_history=True)


    print('value_fn:', value_fn)
    print('policy:', policy)

    plt.figure(1)
    plt.subplot(211)
    label_values = [
        ('1', history[1]),
        ('2', history[2]),
        ('3', history[3]),
        ('final', value_fn),
    ]
    for label, values in label_values:
        plt.plot(range(len(values)), values, label=label)
    plt.legend()
    plt.subplot(212)
    plt.plot(range(len(policy)), policy)
    plt.show()

    policy1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 11, 10, 16, 17, 18, 6, 5, 4, 3, 2, 1, 25, 1, 2, 3, 4, 5, 6, 32, 8, 9, 10, 11, 13, 12, 39, 10, 9, 8, 7, 6, 5, 4, 3, 2, 49, 50, 1, 2, 3, 4, 45, 6, 7, 8, 9, 10, 39, 12, 37, 14, 35, 9, 8, 7, 6, 5, 21, 3, 2, 24, 25, 1, 2, 3, 4, 5, 6, 18, 8, 16, 10, 11, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    policy2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 25, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 50, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 25, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    def play_episode(policy):
        total_reward = 0
        state = env.reset()
        while True:
            action = policy[state]
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                return total_reward


    def evaluate_policy(policy, num_episodes=100000):
        rewards = [ play_episode(policy) for _ in range(num_episodes) ]
        return sum(rewards) / len(rewards)


    policy1_value = evaluate_policy(policy1)
    policy2_value = evaluate_policy(policy2)
    print('policy1 (ours):', policy1_value)
    print('policy2 (optimal):', policy2_value)
