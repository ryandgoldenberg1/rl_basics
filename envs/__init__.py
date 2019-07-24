from gym.envs.registration import register

register(id='GridWorld-v0', entry_point='envs.grid_world:GridWorldEnv')
register(id='WindyGridWorld-v0', entry_point='envs.windy_grid_world:WindyGridWorldEnv')
register(id='GamblersProblem-v0', entry_point='envs.gamblers_problem:GamblersProblemEnv')
register(id='DiscreteBlackJack-v0', entry_point='envs.black_jack_wrapper:BlackJackWrapperEnv')
