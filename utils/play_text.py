import argparse
import gym


def play_text(env_id, keys_to_action=None):
    env = gym.make(env_id)
    if keys_to_action is None and hasattr(env, 'get_keys_to_action'):
        keys_to_action = env.get_keys_to_action()
    env.reset()
    episode_reward = 0
    while True:
        env.render()
        action = input('> ').strip()
        if keys_to_action is not None and action in keys_to_action:
            action = keys_to_action[action]
        else:
            action = int(action)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        print('Reward: {}, Episode: {}'.format(reward, episode_reward))
        if done:
            env.render()
            print('Finished')
            env.reset()
            episode_reward = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', required=True)
    args = parser.parse_args()
    play_text(env_id=args.env_id)


if __name__ == '__main__':
    main()
