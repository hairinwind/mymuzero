import datetime
import os

import gym
import numpy
import torch

from .abstract_game import AbstractGame
from myEnv.TECLStockEnv import TECLCustomEnv
from myEnv.stockDataAlphaReader import countPerDay, symbolCount, signalCount
from myEnv.gameConfig import GameConfig

from games.connect4 import MuZeroConfig as Connect4Config
from games.cartpole import MuZeroConfig as CartpoleConfig


# from rich.console import Console

# console = Console()

class MuZeroConfig:
    def __init__(self):
        self.gameConfig = GameConfig()
        config = self.getNeuralNetworkConfig(self.gameConfig.neural_network_config)   
        self.config = config 

        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = config.seed  # Seed for numpy, torch and the game
        self.max_num_gpus = config.max_num_gpus  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available


        ### Game
        self.observation_shape = (countPerDay, symbolCount, signalCount)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(2))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = config.num_workers  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = config.selfplay_on_gpu
        self.max_moves = self.gameConfig.max_moves  # Maximum number of moves if game is not finished before
        self.num_simulations = self.gameConfig.num_simulations  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = config.temperature_threshold  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = config.root_dirichlet_alpha
        self.root_exploration_fraction = config.root_exploration_fraction

        # UCB formula
        self.pb_c_base = config.pb_c_base
        self.pb_c_init = config.pb_c_init



        ### Network
        self.network = config.network  # "resnet" / "fullyconnected"
        self.support_size = config.support_size  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward))) TODO
        
        # Residual Network
        self.downsample = config.downsample  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = config.blocks  # Number of blocks in the ResNet
        self.channels = config.channels  # Number of channels in the ResNet
        self.reduced_channels_reward = config.reduced_channels_reward  # Number of channels in reward head
        self.reduced_channels_value = config.reduced_channels_value  # Number of channels in value head
        self.reduced_channels_policy = config.reduced_channels_policy  # Number of channels in policy head
        self.resnet_fc_reward_layers = config.resnet_fc_reward_layers  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = config.resnet_fc_value_layers  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = config.resnet_fc_policy_layers  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = config.encoding_size
        self.fc_representation_layers = config.fc_representation_layers  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = config.fc_dynamics_layers  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = config.fc_reward_layers  # Define the hidden layers in the reward network
        self.fc_value_layers = config.fc_value_layers  # Define the hidden layers in the value network
        self.fc_policy_layers = config.fc_policy_layers  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = self.gameConfig.training_steps  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = self.gameConfig.batch_size  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = config.optimizer  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = config.weight_decay  # L2 weights regularization
        self.momentum = config.momentum  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = config.lr_init  # Initial learning rate
        self.lr_decay_rate = config.lr_decay_rate  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = config.lr_decay_steps



        ### Replay Buffer
        self.replay_buffer_size = config.replay_buffer_size  # Number of self-play games to keep in the replay buffer ## TODO increasing this will increase the replay file size?
        self.num_unroll_steps = self.gameConfig.num_unroll_steps  # Number of game moves to keep for every batch element
        self.td_steps = self.gameConfig.td_steps  # Number of steps in the future to take into account for calculating the target value
        self.PER = config.PER  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = config.PER_alpha  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = config.use_last_model_value # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = config.reanalyse_on_gpu



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = config.self_play_delay  # Number of seconds to wait after each played game
        self.training_delay = config.training_delay  # Number of seconds to wait after each training step
        self.ratio = config.ratio  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        return self.config.visit_softmax_temperature_fn(trained_steps)

    def getNeuralNetworkConfig(self, neural_network_config):
        if neural_network_config == 'cartpole':
            print("use cartpole config...")
            return CartpoleConfig()
        if neural_network_config == 'connect4':
            print("use connect4 config...")
            return Connect4Config()
class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        # self.env = gym.make("CartPole-v1")
        self.env = TECLCustomEnv(configFile='teclConfig.json') #, frame_bound=(10,200)
        if seed is not None:
            self.env.seed(seed)

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done, info = self.env.step(action)
        # print("====== step ======")
        # print("action: " + action)
        # print("observation shape: " + observation.shape)
        # print("reward: " + reward)
        # print("info: " + info)
        # console.log("step", log_locals=True)
        return observation, reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.        

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(2))
        # return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        # print("====== reset ======")
        return self.env.reset()

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Sell",
            1: "Buy"
        }
        return f"{action_number}. {actions[action_number]}"
