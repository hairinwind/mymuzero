from myEnv.TECLStockEnv import TECLCustomEnv, Actions, verifyObservation
import numpy as np

def printObservation(observation):
    for index, item in enumerate(observation):
        print(f"----------{index}------------")
        for quote in item:
            print(quote)

if __name__ == "__main__":
    env = TECLCustomEnv(configFile='teclConfig.json') #, frame_bound=(10,200) 
    verifyObservation(env)

    env.reset()
    print(f"before any step, last trade tick: {env._last_trade_tick}")
    print('-----------------')
    observation1, step_reward1, done1, info1 = env.step(Actions.Buy.value)
    print(f"last trade tick: {env._last_trade_tick}")
    assert env._last_trade_tick == 0
    assert step_reward1 == 0

    print('-----------------')
    observation2, step_reward2, done2, info2 = env.step(Actions.Buy.value)
    print(f"last trade tick: {env._last_trade_tick}")
    assert env._last_trade_tick == 0
    assert step_reward2 == 0

    print('-----------------')
    observation3, step_reward3, done3, info3 = env.step(Actions.Sell.value)
    # print(step_reward)
    print(f"last trade tick: {env._last_trade_tick}")
    assert env._last_trade_tick == 2
    assert step_reward3 == env.prices[2] - env.prices[0]

    print('-----------------')
    observation4, step_reward4, done4, info4 = env.step(Actions.Sell.value)
    # print(step_reward)
    print(f"last trade tick: {env._last_trade_tick}")
    assert env._last_trade_tick == 2
    assert step_reward4 == 0

    print('-----------------')
    observation5, step_reward5, done5, info5 = env.step(Actions.Buy.value)
    # print(step_reward)
    print(f"last trade tick: {env._last_trade_tick}")
    assert env._last_trade_tick == 4
    assert step_reward5 == 0

    print('-----------------')
    observation6, step_reward6, done6, info6 = env.step(Actions.Sell.value)
    # print(step_reward)
    print(f"last trade tick: {env._last_trade_tick}")
    assert env._last_trade_tick == 5
    assert step_reward6 == env.prices[5] - env.prices[4], f"expect {env.prices[5] - env.prices[4]}, but was {step_reward6}"
 
    print("manual test done...")