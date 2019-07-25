import gym
import random
import envs


class EligibilityTrace:
    def __init__(self, num_states, num_actions, discount_factor, lambda_):
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.lambda_ = lambda_
        self.reset()

    def get(self, state, action):
        return self.values[state][action]

    def update(self, state, action, action_value_fn):
        max_action_value = max(action_value_fn[state])
        actual_action_value = action_value_fn[state][action]
        if max_action_value == actual_action_value:
            self.decay()
        else:
            self.reset()
        self.values[state][action] += 1

    def decay(self):
        decay_factor = self.discount_factor * self.lambda_
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.values[s][a] *= decay_factor

    def reset(self):
        self.values = [ [0] * self.num_actions for _ in range(self.num_states) ]


class WatkinsQLambda:
    def __init__(self, env):
        self.env = env

    def fit(self, num_episodes, step_size, discount_factor, epsilon, lambda_):
        num_states = self.env.observation_space.n
        num_actions = self.env.action_space.n
        action_value_fn = [ [0] * num_actions for _ in range(num_states) ]
        rewards = []
        for episode_idx in range(num_episodes):
            eligibility_trace = EligibilityTrace(num_states, num_actions, discount_factor, lambda_)
            state = self.env.reset()
            action = self._sample_action(action_value_fn, state, epsilon)
            done = False
            episode_reward = 0
            step = 0
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self._sample_action(action_value_fn, next_state, epsilon)
                max_action = self._max_action(action_value_fn, next_state)
                delta = reward + discount_factor * action_value_fn[next_state][max_action] - action_value_fn[state][action]
                eligibility_trace.update(state, action, action_value_fn)
                for s in range(num_states):
                    for a in range(num_actions):
                        e = eligibility_trace.get(s, a)
                        action_value_fn[s][a] += delta * step_size * e
                state = next_state
                action = next_action
                episode_reward += (discount_factor ** step) * reward
                step += 1
            rewards.append(episode_reward)
            episode_num = episode_idx + 1
            if episode_num % 1000 == 0:
                avg_reward = sum(rewards) / len(rewards)
                print('Episode: {}, Avg Reward: {}'.format(episode_num, avg_reward))
                rewards = []
        return action_value_fn

    def _sample_action(self, action_value_fn, state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        return self._max_action(action_value_fn, state)

    def _max_action(self, action_value_fn, state):
        action_values = action_value_fn[state]
        max_action_value = max(action_values)
        max_actions = [ i for i, value in enumerate(action_values) if value == max_action_value ]
        return random.choice(max_actions)


if __name__ == '__main__':
    num_episodes = 10000
    step_size = 0.1
    discount_factor = 1.
    epsilon = 0.1
    lambda_ = 0.9
    env_id = 'DiscreteBlackJack-v0'

    env = gym.make(env_id)
    q_lambda = WatkinsQLambda(env)
    q_lambda.fit(num_episodes=num_episodes, step_size=step_size, discount_factor=discount_factor, epsilon=epsilon, lambda_=lambda_)
