from gym.envs.registration import register

register(id='GridWorld-v0', entry_point='envs.grid_world:GridWorldEnv')
register(id='GamblersProblem-v0', entry_point='envs.gamblers_problem:GamblersProblemEnv')
