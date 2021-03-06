import gym
import json
import numpy as np
import os
import pickle

from enum import Enum
from gym import spaces
from matplotlib import pyplot as plt
# from myEnv.stockDataReader import readToDataFrame, getArray, countPerDay, targetSymbolIndex
from myEnv.stockDataAlphaReader import readToDataFrame, getArray, countPerDay
from util import debug

def readConfig(config): 
    # print(f'config is {config}')
    with open(os.path.join('data/working', config), 'r') as f:
        configJson = json.load(f)
    # print(configJson)
    return configJson

class Actions(Enum):
    Sell = 0
    Buy = 1
    # Hold = 2
class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long

class TECLCustomEnv(gym.Env):
    def __init__(self, configFile):
        self.config = readConfig(configFile)
        print(f"read data between {self.config['startDate']} and {self.config['endDate']}")
        self.df = readToDataFrame(self.config['startDate'], self.config['endDate'])
        # assert len(frame_bound) == 2

        # self.frame_bound = frame_bound
        
        # super().__init__(df, window_size
        # assert df.ndim == 2
        self.seed()
        # self.df = df
        # self.window_size = window_size
        # self.prices, self.signal_features = self._process_data()
        # self.shape = (window_size * self.signal_features.shape[1],)
        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        ## self.observation_space = spaces.Discrete(60)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)
        # episode
        # self._start_tick = self.window_size
        # self._end_tick = len(self.prices) - 1
        
        # self.prices = self.df[self.df['symbol'] == 'TECL']['regularMarketPrice'].to_numpy()
        prices = self.df[self.df['symbol'] == 'TECL']['close'].to_numpy()
        self.prices = prices[countPerDay - 1:]
        print("init self.prices: ", self.prices, len(self.prices))
        
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
        self._current_tick = 0
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short # TODO try initial position is Long
        # self._position_history = (self.window_size * [None]) + [self._position]
        self._position_history = []
        self._total_reward = self.prices[countPerDay]
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
        # ?????????????????????????????????????????????????????? day ???????????????10
        # ?????????????????????????????????????????????????????? ????????????5?????????????????????5
        # ????????????????????????????????????????????? open/close/high/low/volume????????????5?????????
        observation = getArray(self.df, self._current_tick)
        # if observation is not None:
        #     assert observation[0][98][0] == self.prices[self._current_tick], f"tick: {self._current_tick}, price: {self.prices[self._current_tick]}, observation: {observation[0][98][0]}" # TECL is @98
        return observation

    def _calculate_reward(self, action):
        return self._calculate_reward_1(action)

    """
    this is the algorithm from anytrading
    """
    def _calculate_reward_1(self, action):
        step_reward = 0

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        # print("trade:", trade)

        if trade:
            current_price = self.prices[self._current_tick]
            # print(f"TECL price {current_price} and current_tick {self._current_tick}")
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                step_reward += price_diff

            # reward ???????????????????????????????????????
            # print("current_price: ", current_price)
            # print("last_trade_price: ", last_trade_price)
            # print("price_diff: ", price_diff)
            # print("step_reward", step_reward)
        
        return step_reward

    def _calculate_total_value(self, action):
        current_price = self.prices[self._current_tick - 1]
        if action == Actions.Sell.value and self._position == Positions.Long:
            my_cash_balance = self.my_shares * current_price * (1 - self.trade_fee_bid_percent)
            my_shares = 0
        elif action == Actions.Buy.value and self._position == Positions.Short:
            my_shares = self.my_cash_balance * (1 - self.trade_fee_ask_percent) / current_price
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
        window_ticks = [0] * len(self._position_history)
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
        if self._done:
            window_ticks = np.arange(len(self._position_history))
            short_ticks = []
            long_ticks = []
            
            # print("self._position_history:", self._position_history)
            # print("window_ticks:", window_ticks)
            for i, tick in enumerate(window_ticks):
                if i == 0 :
                    previousPosition = Positions.Short
                else:
                    previousPosition = self._position_history[i-1]

                if self._position_history[i] == Positions.Short and  previousPosition == Positions.Long:
                    short_ticks.append(tick)
                    print("add short_ticks:", tick)
                if self._position_history[i] == Positions.Long and previousPosition == Positions.Short: 
                    long_ticks.append(tick)
                    print("add long_ticks:", tick)
                
            result = {}
            result['prices'] = self.prices
            result['history'] = self.history
            result['total_value_history'] = self.my_total_value_history
            result['short_ticks'] = short_ticks
            result['long_ticks'] = long_ticks
            result['total_reward'] = self._total_reward
            result['total_profit'] = self._total_profit
        
            # print("result: ", result)
            print("done, picle the file...")
            with open('evaluate/result.pkl', 'wb') as file:
                pickle.dump(result, file)

    def close(self):
        plt.close()

    def legal_actions(self):
        # depend on countPerDay, max transaction per day shall be 2
        # if self._position == Positions.Short: 
        #     return [Actions.Buy.value, Actions.Hold.value]
        # if self._position == Positions.Long:
        #     return [Actions.Sell.value, Actions.Hold.value]
        return [Actions.Sell.value, Actions.Buy.value]

    def step(self, action):
        # print(f"action: {Actions(action).name}, tick {self._current_tick}")
        self._done = False
        self._current_tick += 1

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        # print(f"tick {self._current_tick}")
        observation = self._get_observation()
        step_reward = 0
        if observation is None:
            # print('observation is None, set self._done to True...')
            self._done = True
        else:
            # print(f"price in observateion: {observation[countPerDay-1][98][0]}")
            # print(f"price {self.prices[self._current_tick]}")
            assert observation[countPerDay-1][0][0] == self.prices[self._current_tick] #98 if full symbols
            step_reward = self._calculate_reward(action)
            # print(f"step_reward {step_reward}")
            self._total_reward += step_reward

        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick
            # print(f"reward {step_reward}")

        
        my_cash_balance, my_shares, my_total_value = self._calculate_total_value(action)
        # print(f"my_total_value {my_total_value}")
        # previous_total_value = self.previousTotalValue()
        self.my_total_value_history.append(my_total_value)
        # step_reward = my_total_value / previous_total_value
        # self._total_reward = my_total_value - self.my_init_cash_balance

        self._position_history.append(self._position)
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position.value
        )
        self._update_history(info)
        return observation, step_reward, self._done, info

    def previousTotalValue(self):
        try:
            previous_total_value = self.my_total_value_history[-1]
        except IndexError:
            previous_total_value = self.my_init_cash_balance
        return previous_total_value

def verifyObservation(env):
    env.reset()
    prices = env.prices
    df = env.df
    count = 0
    while True:
        observation, step_reward, done, info = env.step(Actions.Buy)
        if observation is None:
            break
        assert observation[0][98][0] == prices[count], f"observation is not correct @{count}"
        count += 1
    print(f"observation is valid... iterate time {count}")
    env.reset()

if __name__ == "__main__":
    env = TECLCustomEnv(configFile='teclConfig.json') #, frame_bound=(10,200) 
    env.reset()

    print("manual test done...")