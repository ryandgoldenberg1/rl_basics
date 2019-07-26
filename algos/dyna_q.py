import gym
import random
from tqdm import trange
import envs




class TabularModel:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.experience = [ [ [] for _ in range(num_actions) ] for _ in range(num_states) ]
        self.seen_states = set()
        self.seen_state_actions = [ set() for _ in range(num_states) ]

    def update(self, state, action, reward, next_state):
        self.seen_states.add(state)
        self.seen_state_actions[state].add(action)
        self.experience[state][action].append((next_state, reward))

    def sample(self):
        state = random.choice(list(self.seen_states))
        action = random.choice(list(self.seen_state_actions[state]))
        next_state, reward = random.choice(self.experience[state][action])
        return state, action, reward, next_state


class DynaQ:
    def __init__(self, env):
        self.env = env

    def fit(self, num_episodes, discount_factor, step_size, epsilon, num_planning_updates):
        num_states = self.env.observation_space.n
        num_actions = self.env.action_space.n
        action_value_fn = [ [0] * num_actions for _ in range(num_states) ]
        model = TabularModel(num_states, num_actions)
        episode_lengths = []
        for episode_idx in trange(num_episodes):
            state = self.env.reset()
            done = False
            step = 0
            while not done:
                action = self.act(action_value_fn, state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q(action_value_fn, state, action, reward, next_state, step_size, discount_factor)
                model.update(state, action, reward, next_state)
                for _ in range(num_planning_updates):
                    s, a, r, ns = model.sample()
                    self.update_q(action_value_fn, s, a, r, ns, step_size, discount_factor)
                state = next_state
                step += 1
            episode_lengths.append(step)
        return action_value_fn, episode_lengths


    def act(self, action_value_fn, state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        action_values = action_value_fn[state]
        max_action_value = max(action_values)
        max_actions = [ i for i, value in enumerate(action_values) if value == max_action_value ]
        return random.choice(max_actions)

    def update_q(self, action_value_fn, state, action, reward, next_state, step_size, discount_factor):
        delta = reward + discount_factor * max(action_value_fn[next_state]) - action_value_fn[state][action]
        action_value_fn[state][action] += step_size * delta


if __name__ == '__main__':
    num_episodes = 100
    discount_factor = 0.95
    step_size = 0.1
    epsilon = 0.1
    num_planning_updates = 5

    env = gym.make('DynaMaze-v0')
    dyna_q = DynaQ(env)
    _, episode_lengths = dyna_q.fit(num_episodes=num_episodes, discount_factor=discount_factor, step_size=step_size,
                                    epsilon=epsilon, num_planning_updates=num_planning_updates)
    print(episode_lengths)
