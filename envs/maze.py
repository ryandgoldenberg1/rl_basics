from gym.envs.toy_text import discrete
from utils.play_text import play_text


class DynaMazeEnv(discrete.DiscreteEnv):
    def __init__(self):
        num_rows = 6
        num_cols = 9
        start_pos = (2, 0)
        goal_pos = (0, 8)
        obstacles = set([
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 5),
            (0, 7),
            (1, 7),
            (2, 7),
        ])

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.start_state = self._pos_to_state(start_pos)
        self.goal_state = self._pos_to_state(goal_pos)
        self.obstacle_states = set(self._pos_to_state(x) for x in obstacles)

        nS = num_rows * num_cols
        nA = 4
        isd = [0] * nS
        isd[self.start_state] = 1

        action_to_delta = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}
        P = {} # P[s][a] = [ (prob, nextstate, reward, done) ]
        for s in range(nS):
            P[s] = {}
            row = int(s / num_cols)
            col = s % num_cols
            for a in range(nA):
                if s in self.obstacle_states:
                    P[s][a] = []
                else:
                    row_delta, col_delta = action_to_delta[a]
                    next_row = row + row_delta
                    next_row = max(next_row, 0)
                    next_row = min(next_row, num_rows - 1)
                    next_col = col + col_delta
                    next_col = max(next_col, 0)
                    next_col = min(next_col, num_cols - 1)
                    next_state = next_row * num_cols + next_col
                    if next_state in self.obstacle_states:
                        next_state = s
                    done = (next_state == self.goal_state)
                    reward = 1 if done else 0
                    P[s][a] = [ (1., next_state, reward, done) ]
        super().__init__(nS=nS, nA=nA, P=P, isd=isd)

    def _pos_to_state(self, pos):
        x, y = pos
        return x * self.num_cols + y

    def _state_to_pos(self, state):
        return (int(state / self.num_cols), state % self.num_cols)

    def render(self, mode='human'):
        grid = [['o' for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        for s in self.obstacle_states:
            row, col = self._state_to_pos(s)
            grid[row][col] = 'X'
        start_row, start_col = self._state_to_pos(self.start_state)
        grid[start_row][start_col] = 'S'
        goal_row, goal_col = self._state_to_pos(self.goal_state)
        grid[goal_row][goal_col] = 'G'
        row, col = self._state_to_pos(self.s)
        grid[row][col] = 'A'
        grid_output = ''.join([ ' '.join(x) + '\n' for x in grid ])
        action_to_text = {0: 'Up', 1: 'Down', 2: 'Right', 3: 'Left'}
        last_action = action_to_text.get(self.lastaction) or 'None'
        output = '(' + last_action + ')\n' + grid_output
        print(output)


    def get_keys_to_action(self):
        return {'u': 0, 'd': 1, 'r': 2, 'l': 3}


if __name__ == '__main__':
    play_text('DynaMaze-v0')
