from gym.envs.toy_text import discrete
from utils.play_text import play_text


class WindyGridWorldEnv(discrete.DiscreteEnv):
    def __init__(self):
        num_rows = 7
        num_cols = 10
        start_state = 30
        end_state = 37
        winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        nS = num_rows * num_cols
        nA = 4
        isd = [0] * nS
        isd[start_state] = 1.
        action_to_delta = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, 1),
            3: (0, -1),
        }
        # P[s][a] = [ (prob, nextstate, reward, done) ]
        P = {}
        for s in range(nS):
            P[s] = {}
            row = int(s / num_cols)
            col = s % num_cols
            wind = winds[col]
            for a in range(nA):
                delta = action_to_delta[a]
                next_row = row + delta[0] - wind
                next_row = max(next_row, 0)
                next_row = min(next_row, num_rows - 1)
                next_col = col + delta[1]
                next_col = max(next_col, 0)
                next_col = min(next_col, num_cols - 1)
                next_state = next_row * num_cols + next_col
                done = (next_state == end_state)
                reward = 0 if done else -1
                P[s][a] = [ (1., next_state, reward, done) ]

        super().__init__(nS=nS, nA=nA, P=P, isd=isd)
        self.winds = winds
        self.start_state = start_state
        self.end_state = end_state
        self.num_rows = num_rows
        self.num_cols = num_cols

    def render(self, mode='human'):
        grid = [[' '] * self.num_cols for _ in range(self.num_rows)]
        start_row, start_col = self._state_to_coords(self.start_state)
        end_row, end_col = self._state_to_coords(self.end_state)
        curr_row, curr_col = self._state_to_coords(self.s)
        grid[start_row][start_col] = 'S'
        grid[end_row][end_col] = 'G'
        grid[curr_row][curr_col] = 'X'

        row_sep = '-' * (1 + 2 * self.num_cols) + '\n'
        col_sep = '|'
        row_strs = []
        for row in grid:
            row_str = col_sep + col_sep.join(row) + col_sep + '\n'
            row_strs.append(row_str)
        grid_output = row_sep + row_sep.join(row_strs) + row_sep
        wind_output = ' ' + ' '.join([ str(x) for x in self.winds ]) + ' \n'

        action_to_text = {0: 'Up', 1: 'Down', 2: 'Right', 3: 'Left'}
        last_action = action_to_text.get(self.lastaction) or 'None'
        output = '(' + last_action + ')\n' + grid_output + wind_output
        print(output)

    def get_keys_to_action(self):
        return {'u': 0, 'd': 1, 'r': 2, 'l': 3}

    def _state_to_coords(self, state):
        row = int(state / self.num_cols)
        col = state % self.num_cols
        return (row, col)


if __name__ == '__main__':
    play_text(env_id='WindyGridWorld-v0')
