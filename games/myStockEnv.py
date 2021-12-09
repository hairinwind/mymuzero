import numpy as np
import pandas as pd
import gym
import gym_anytrading

from enum import Enum
from gym import spaces
from matplotlib import pyplot as plt

# Stable baselines - rl stuff
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

class Actions(Enum):
    Sell = 0
    Buy = 1
    Hold = 2

class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long

class MyCustomEnv(gym.Env):
    def __init__(self, df, window_size, frame_bound):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        
        # super().__init__(df, window_size)
        assert df.ndim == 2
        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size * self.signal_features.shape[1],)
        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        # self.observation_space = spaces.Discrete(60)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)
        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size):self._current_tick].flatten()

    def _process_data(self):
        # print("..._process_data...")
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        prices = self.df.loc[:, 'Close'].to_numpy()[start:end]
        signal_features = self.df.loc[:, ['DateNumber', 'Open', 'Close','High', 'Low', 'Volume']].to_numpy()[start:end]
        return prices, signal_features
        # np.ndarray.flatten(signal_features)

    def _calculate_reward(self, action):
        step_reward = 0

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        # print("trade:", trade)

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                step_reward += price_diff

            # print("current_price: ", current_price)
            # print("last_trade_price: ", last_trade_price)
            # print("price_diff: ", price_diff)
            # print("step_reward", step_reward)

        return step_reward
    
    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price
                # print("_update_profit::shares: ", shares)
                # print("_update_profit::_total_profit", self._total_profit)
    
    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)
            
        # print("_update_history::history: ", self.history)

    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        hold_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )
        
    def render(self):
        print("=== render ===")
        
    def close(self):
        plt.close()

    def step(self, action):
        # print("action: ", Actions(action).name)
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        # print("trade? ", trade)

        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick
            # print("_position:", self._position)
            # print("self._last_trade_tick: ", self._last_trade_tick)

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info


def getData(): 
    df = pd.read_csv('data/gmedata.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['DateNumber'] = df['Date'].apply(lambda x: int(x.strftime('%Y%m%d')))
    df.sort_values('Date', ascending=True, inplace=True)
    df.set_index('Date', inplace=True)
    # df.dtypes
    # df.head(15)
    return df

def learn(df):
    env = MyCustomEnv(df=df, window_size=12, frame_bound=(12,200))
    # print("env shape:", env.shape)

    # env.reset()
    # observation, step_reward, done, info = env.step(1)

    # print(info)
    # print("step_reward:", step_reward)

    # observation, step_reward, done, info = env.step(0)
    # print(info)
    # print("step_reward:", step_reward)

    # observation, step_reward, done, info = env.step(1)
    # print(info)
    # print("step_reward:", step_reward)

    env.reset()
    env_maker = lambda: env
    myenv = DummyVecEnv([env_maker])

    model = A2C('MlpLstmPolicy', myenv, verbose=1) 
    model.learn(total_timesteps=10000)
    return model

def evaluate(model):
    ## Evaluation
    env = MyCustomEnv(df=df, window_size=12, frame_bound=(201,250))
    obs = env.reset()
    while True: 
        obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            print("info", info)
            break

    plt.figure(figsize=(15,6))
    plt.cla()
    env.render_all()
    plt.show()