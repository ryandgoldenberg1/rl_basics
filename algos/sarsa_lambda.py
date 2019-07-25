import random
import gym
import envs


class EligibilityTrace:
    def __init__(self, num_states, num_actions, discount_factor, lambda_):
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.lambda_ = lambda_
        self.values = [ [0] * num_actions for _ in range(num_states) ]

    def update(self, state, action):
        self.values[state][action] += 1

    def decay(self):
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.values[s][a] *= self.discount_factor * self.lambda_

    def get(self, state, action):
        return self.values[state][action]


class SarsaLambda:
    def __init__(self, env):
        self.env = env

    def fit(self, discount_factor, step_size, epsilon, num_episodes, lambda_):
        num_states = self.env.observation_space.n
        num_actions = self.env.action_space.n
        action_value_fn = [ [0] * num_actions for _ in range(num_states) ]
        rewards = []
        for episode_idx in range(num_episodes):
            state = self.env.reset()
            done = False
            eligibility_trace = EligibilityTrace(num_states, num_actions, discount_factor, lambda_)
            action = self._sample_action(action_value_fn, state, epsilon)
            episode_reward = 0
            step = 0
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self._sample_action(action_value_fn, next_state, epsilon)
                delta = reward + discount_factor * action_value_fn[next_state][next_action] - action_value_fn[state][action]
                eligibility_trace.update(state, action)
                for s in range(num_states):
                    for a in range(num_actions):
                        e = eligibility_trace.get(s, a)
                        action_value_fn[s][a] += delta * e * step_size
                eligibility_trace.decay()
                state = next_state
                action = next_action
                episode_reward += (discount_factor ** step) * reward
                step += 1
            rewards.append(episode_reward)
            if (episode_idx + 1) % 1000 == 0:
                avg_reward = sum(rewards) / len(rewards)
                print('Episode: {}, Avg Reward: {}'.format(episode_idx + 1, avg_reward))
                rewards = []
        return action_value_fn

    def _sample_action(self, action_value_fn, state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        action_values = action_value_fn[state]
        max_action_value = max(action_values)
        max_actions = [ i for i, value in enumerate(action_values) if value == max_action_value ]
        return random.choice(max_actions)



if __name__ == '__main__':
    num_episodes = 100000
    step_size = 0.1
    epsilon = 0.1
    lambda_ = 0.9
    discount_factor = 1.
    env_id = 'WindyGridWorld-v0'

    env = gym.make(env_id)
    agent = SarsaLambda(env)
    agent.fit(discount_factor=discount_factor, step_size=step_size, epsilon=epsilon, num_episodes=num_episodes, lambda_=lambda_)
