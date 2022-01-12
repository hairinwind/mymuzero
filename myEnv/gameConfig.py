
class GameConfig:
    def __init__(self):
        self.neural_network_config = 'connect4' # cartpole connect4 
        self.max_moves = 500
        self.num_simulations = 10  # Number of future moves self-simulated
        self.training_steps = 1000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 16  # Number of parts of games to train on at each training step
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.td_steps = 10  # Number of steps in the future to take into account for calculating the target value