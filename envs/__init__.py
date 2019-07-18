from gym.envs.registration import register
from .grid_world import GridWorldEnv

register(id='GridWorld-v0', entry_point='envs.grid_world:GridWorldEnv')
