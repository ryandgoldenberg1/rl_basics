import numpy as np
import random


def sample_action(policy, state, epsilon_greedy=False, num_actions=None, epsilon=None):
    if isinstance(policy, dict) and state not in policy:
        raise ValueError('policy does not have state: {}'.format(state))
    distribution = policy[state]
    if epsilon_greedy:
        if epsilon is None:
            raise ValueError('Epsilon must be supplied')
        if num_actions is None:
            raise ValueError('num_actions must be supplied if epsilon_greedy')
        if random.random() < epsilon:
            return random.choice(range(num_actions))
    if isinstance(distribution, int):
        return distribution
    return np.random.choice(len(distribution), p=distribution)


def generate_episode(env, policy, epsilon_greedy=False, epsilon=None):
    history = []
    state = env.reset()
    num_actions = env.action_space.n
    while True:
        action = sample_action(policy, state, epsilon_greedy=epsilon_greedy, epsilon=epsilon, num_actions=num_actions)
        next_state, reward, done, _ = env.step(action)
        history.append({ 'state': state, 'action': action, 'reward': reward, 'next_state': next_state })
        if done:
            return history
        state = next_state
