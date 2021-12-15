import pickle
from matplotlib import pyplot as plt

with open('evaluate/result.pkl', 'rb') as evaluateResultFile:
    result = pickle.load(evaluateResultFile)
    
    # print("result: ", result)

    plt.figure(figsize=(15,6))
    plt.cla()
    plt.plot(result['prices'])
    plt.plot(result['short_ticks'], result['prices'][result['short_ticks']], 'ro')
    plt.plot(result['long_ticks'], result['prices'][result['long_ticks']], 'go')
    plt.plot(result['hold_ticks'], result['prices'][result['hold_ticks']], 'yo')
    plt.suptitle(
            "Total Reward: %.6f" % result['total_reward'] + ' ~ ' +
            "Total Profit: %.6f" % result['total_profit']
        )
    plt.show()
