import numpy as np
import pandas as pd
import gym
# import gym_anytrading
import os
import pickle

from enum import Enum
from gym import spaces
from matplotlib import pyplot as plt

# Stable baselines - rl stuff
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import A2C

class Actions(Enum):
    Sell = 0
    Buy = 1
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
        ## self.observation_space = spaces.Discrete(60)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)
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

        self.my_init_cash_balance = 1
        self.my_cash_balance = self.my_init_cash_balance
        self.my_shares = 0
        self.my_total_value_history = []

    def reset(self):
        # print("=== reset === ")
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        self.my_cash_balance = self.my_init_cash_balance
        self.my_shares = 0
        self.my_total_value_history = []
        return self._get_observation()

    def _get_observation(self):
        # return self.signal_features[(self._current_tick-self.window_size):self._current_tick].flatten()
        # observation[days][stocks][signals]
        # 第一维度代表多少天的数据，比如取两周 day 数据，就是10
        # 第二维度代表有多少个参考的股票代码， 比如参考5个股票，那就是5
        # 第三维度代表股票具体指标，比如 open/close/high/low/volume，那就是5个参数
        data = self.signal_features[(self._current_tick-self.window_size):self._current_tick]
        observation = np.array([np.array([item]) for item in data])
        return observation

    def _process_data(self):
        # print("..._process_data...")
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        prices = self.df.loc[:, 'Close'].to_numpy()[start:end]
        signal_features = self.df.loc[:, ['Open', 'Close', 'High', 'Low', 'Volume', 'SMA12', 'SMA20', 'RSI', 'OBV']].to_numpy()[start:end]
        return prices, signal_features
        # np.ndarray.flatten(signal_features)

    def _calculate_reward(self, action):
        step_reward = 0

        # trade = False
        # if ((action == Actions.Buy.value and self._position == Positions.Short) or
        #     (action == Actions.Sell.value and self._position == Positions.Long)):
        #     trade = True

        # # print("trade:", trade)

        # if trade:
        #     current_price = self.prices[self._current_tick]
        #     last_trade_price = self.prices[self._last_trade_tick]
        #     price_diff = current_price - last_trade_price

        #     if self._position == Positions.Long:
        #         step_reward += price_diff

        #     # reward 是买卖时候每股股票的价格差
        #     # print("current_price: ", current_price)
        #     # print("last_trade_price: ", last_trade_price)
        #     # print("price_diff: ", price_diff)
        #     # print("step_reward", step_reward)

        # return self.my_cash_balance, self.my_shares, self.my_total_value
        current_price = self.prices[self._current_tick]
        if action == Actions.Sell.value and self._position == Positions.Long:
            my_cash_balance = self.my_shares * current_price
            my_shares = 0
        elif action == Actions.Buy.value and self._position == Positions.Short:
            my_shares = self.my_cash_balance / current_price
            my_cash_balance = 0
        else: 
            my_cash_balance = self.my_cash_balance
            my_shares = self.my_shares
        my_total_value = my_shares * current_price + my_cash_balance
        return my_cash_balance, my_shares, my_total_value
            

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
            if self._position_history[i] == Positions.Short and self._position_history[i-1] == Positions.Long:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long and self._position_history[i-1] == Positions.Short: 
                long_ticks.append(tick)
            else: 
                hold_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )
        
    def render(self):
        window_ticks = np.arange(len(self._position_history))
        short_ticks = []
        long_ticks = []
        hold_ticks = []
        for i, tick in enumerate(window_ticks):
            if i == 0 :
                previousPosition = Positions.Short
            else:
                previousPosition = self._position_history[i-1]

            if self._position_history[i] == Positions.Short and  previousPosition == Positions.Long:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long and previousPosition == Positions.Short: 
                long_ticks.append(tick)
            else: 
                hold_ticks.append(tick)

        # print("render prices: ", self.prices)
        # print("short_ticks:", short_ticks)
        # print("long_ticks: ", long_ticks)

        result = {}
        result['prices'] = self.prices
        result['total_value_history'] = self.my_total_value_history
        result['short_ticks'] = short_ticks
        result['long_ticks'] = long_ticks
        result['hold_ticks'] = hold_ticks
        result['total_reward'] = self._total_reward
        result['total_profit'] = self._total_profit
        
        # print("result: ", result)
        with open('evaluate/result.pkl', 'wb') as file:
            pickle.dump(result, file)

    def close(self):
        plt.close()

    # def legal_actions(self):
    #     if self._position == Positions.Short: 
    #         return [Actions.Buy.value, Actions.Hold.value]
    #     if self._position == Positions.Long:
    #         return [Actions.Sell.value, Actions.Hold.value]

    def step(self, action):
        # print("action: ", Actions(action).name)
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        # step_reward = self._calculate_reward(action)
        # self._total_reward += step_reward
        # self._update_profit(action)
        self.my_cash_balance, self.my_shares, my_total_value = self._calculate_reward(action)
        previous_total_value = self.previousTotalValue()
        self.my_total_value_history.append(my_total_value)
        step_reward = my_total_value / previous_total_value
        self._total_reward = my_total_value - self.my_init_cash_balance
        # print("action:{} cash_balance:{}, shares:{}, totalValue:{}, step_reward:{}".format(action, self.my_cash_balance, self.my_shares, my_total_value, step_reward), end='\r')


        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

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

        # return observation, step_reward, self._done, info

        # if my_total_value > 6:
        #     self._done = True
        #     print("... my_total_value > 5 ... set sefl._done to True")
        return observation, step_reward, self._done, info

    def previousTotalValue(self):
        try:
            previous_total_value = self.my_total_value_history[-1]
        except IndexError:
            previous_total_value = self.my_init_cash_balance
        return previous_total_value


def getData(): 
    target_data_file = 'data/data.csv'
    df = pd.read_csv(target_data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df['DateNumber'] = df['Date'].apply(lambda x: int(x.strftime('%Y%m%d')))
    df.sort_values('Date', ascending=True, inplace=True)
    df.set_index('Date', inplace=True)
    # df.dtypes
    print(df.head(15))
    return df

# def learn(df):
#     env = MyCustomEnv(df=df, window_size=12, frame_bound=(12,200))
#     # print("env shape:", env.shape)

#     # env.reset()
#     # observation, step_reward, done, info = env.step(1)

#     # print(info)
#     # print("step_reward:", step_reward)

#     # observation, step_reward, done, info = env.step(0)
#     # print(info)
#     # print("step_reward:", step_reward)

#     # observation, step_reward, done, info = env.step(1)
#     # print(info)
#     # print("step_reward:", step_reward)

#     env.reset()
#     env_maker = lambda: env
#     myenv = DummyVecEnv([env_maker])

#     model = A2C('MlpLstmPolicy', myenv, verbose=1) 
#     model.learn(total_timesteps=10000)
#     return model

# def evaluate(model):
#     ## Evaluation
#     env = MyCustomEnv(df=df, window_size=12, frame_bound=(201,250))
#     obs = env.reset()
#     while True: 
#         obs = obs[np.newaxis, ...]
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)
#         if done:
#             print("info", info)
#             break

#     plt.figure(figsize=(15,6))
#     plt.cla()
#     env.render_all()
#     plt.show()

if __name__ == "__main__":
    df = getData()
    print("data size", df.size)

    env = MyCustomEnv(df=df, window_size=12, frame_bound=(12,200))
    prices, signal_features = env._process_data()
    print("signal_features", signal_features)

    env.reset()

    observation = env._get_observation()
    print(observation.shape)
    print(type(observation))
    print(observation)

    print("manual test done...")