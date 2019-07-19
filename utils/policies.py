import numpy as np


def sample_action(policy, state):
    if isinstance(policy, dict) and state not in policy:
        raise ValueError('policy does not have state: {}'.format(state))
    distribution = policy[state]
    if isinstance(distribution, int):
        return distribution
    return np.random.choice(len(distribution), p=distribution)


def generate_episode(env, policy):
    history = []
    state = env.reset()
    while True:
        action = sample_action(policy, state)
        next_state, reward, done, _ = env.step(action)
        history.append({ 'state': state, 'action': action, 'reward': reward, 'next_state': next_state })
        if done:
            return history
        state = next_state
