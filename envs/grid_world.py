from gym.envs.toy_text import discrete
from utils.play_text import play_text


# Environment from Barto & Sutton Example 4.1
class GridWorldEnv(discrete.DiscreteEnv):
    def __init__(self):
        nS = 15
        nA = 4 # { up, down, right, left }
        # P[s][a] = [ (prob, nextstate, reward, done) ]
        P = {}
        P[0] = {a: [] for a in range(nA)}
        for s in range(1, nS):
            P[s] = {}
            # up, 0
            up_dest = s - 4
            if up_dest < 0:
                up_dest = s
            up_reward = -1
            up_done = (up_dest == 0)
            P[s][0] = [ (1., up_dest, up_reward, up_done) ]

            # down, 1
            down_dest = s + 4
            if down_dest > 15:
                down_dest = s
            down_dest = down_dest % 15
            down_reward = -1
            down_done = (down_dest == 0)
            P[s][1] = [ (1., down_dest, down_reward, down_done) ]

            # right, 2
            right_dest = s + 1
            if right_dest % 4 == 0:
                right_dest = s
            right_dest = right_dest % 15
            right_reward = -1
            right_done = (right_dest == 0)
            P[s][2] = [ (1., right_dest, right_reward, right_done) ]

            # left, 3
            left_dest = s - 1
            if s % 4 == 0:
                left_dest = s
            left_reward = -1
            left_done = (left_dest == 0)
            P[s][3] = [ (1., left_dest, left_reward, left_done) ]
        isd = [ 1.0 / 14 ] * 15
        isd[0] = 0.0 # terminal state
        super().__init__(nS=nS, nA=nA, P=P, isd=isd)

    def render(self, mode='human'):
        grid = [[' '] * 4 for _ in range(4)]
        grid[0][0] = 'G'
        grid[-1][-1] = 'G'
        row = int(self.s / 4)
        col = self.s % 4
        grid[row][col] = 'X'

        row_sep = '-' * 9 + '\n'
        col_sep = '|'
        row_strs = []
        for row in grid:
            row_str = col_sep + col_sep.join(row) + col_sep + '\n'
            row_strs.append(row_str)
        grid_output = row_sep + row_sep.join(row_strs) + row_sep

        action_to_text = {0: 'Up', 1: 'Down', 2: 'Right', 3: 'Left'}
        last_action = action_to_text.get(self.lastaction) or 'None'

        output = '(' + last_action + ')\n' + grid_output
        print(output)

    def get_keys_to_action(self):
        return {'u': 0, 'd': 1, 'r': 2, 'l': 3}


if __name__ == '__main__':
    play_text(env_id='GridWorld-v0')
