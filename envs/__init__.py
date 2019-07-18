from gym.envs.registration import register

register(id='GridWorld-v0', entry_point='envs.grid_world:GridWorldEnv')
register(id='CarRental-v0', entry_point='envs.car_rental:CarRentalEnv')
